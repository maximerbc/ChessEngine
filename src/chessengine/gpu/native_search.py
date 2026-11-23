"""Compact legal edges for the same deterministic uniform-prior PUCT algorithm.

PGX labels remain canonical. Capacity is checked, never a move shortlist: a
position exceeding it fails the tournament instead of dropping legal actions.
Historical dense search is retained as an independent parity reference.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import mctx

from . import rules
from .search import evaluate, selection

CAPACITY = 256


def legal_edges(state, done):
    mask = state.game.legal_action_mask
    labels = jax.vmap(lambda x: jnp.nonzero(x, size=CAPACITY, fill_value=0)[0])(mask)
    counts = mask.sum(1)
    valid = jnp.arange(CAPACITY)[None, :] < counts[:, None]
    valid = jnp.where(done[:, None], jax.nn.one_hot(jnp.zeros(len(done), jnp.int32), CAPACITY, dtype=bool), valid)
    return labels, valid, counts > CAPACITY


@lru_cache(maxsize=8)
def make_search(simulations):
    if simulations < 1:
        raise ValueError('Positive simulation budget required')

    @jax.jit
    def search(params, states, kinds):
        done, terminal_values = jax.vmap(rules.terminal)(states)
        labels, masks, too_many = legal_edges(states, done)
        states = states._replace(overflow=states.overflow | too_many)
        values = jnp.where(done, terminal_values, jax.vmap(evaluate)(params, states, kinds))
        root = mctx.RootFnOutput(prior_logits=jnp.where(masks, 0., -jnp.inf),
            value=values, embedding=(states, kinds, labels))

        def recurrent(parameters, _key, edges, embeddings):
            before, evaluator_kinds, mapping = embeddings
            was_done, was_value = jax.vmap(rules.terminal)(before)
            actions = jnp.take_along_axis(mapping, edges[:, None], 1)[:, 0]
            after = jax.vmap(lambda s, a, d: jax.lax.cond(d, lambda _: s,
                lambda _: rules.step(s, a), None))(before, actions, was_done)
            ended, terminal_value = jax.vmap(rules.terminal)(after)
            labels, mask, too_many = legal_edges(after, ended)
            after = after._replace(overflow=after.overflow | too_many)
            value = jnp.where(ended, 0., jax.vmap(evaluate)(parameters, after, evaluator_kinds))
            output = mctx.RecurrentFnOutput(reward=jnp.where(was_done, was_value, jnp.where(ended, -terminal_value, 0.)),
                discount=jnp.where(ended | was_done, 0., -1.), prior_logits=jnp.where(mask, 0., -jnp.inf), value=value)
            return output, (after, evaluator_kinds, labels)

        tree = mctx.search(params, jax.random.PRNGKey(0), root=root, recurrent_fn=recurrent,
            root_action_selection_fn=selection, interior_action_selection_fn=selection,
            num_simulations=simulations, invalid_actions=~masks)
        visits = tree.children_visits[:, 0]
        edges = jnp.argmax(visits, axis=1).astype(jnp.int32)
        actions = jnp.take_along_axis(labels, edges[:, None], 1)[:, 0]
        q = tree.children_rewards[:, 0] + tree.children_discounts[:, 0] * tree.children_values[:, 0]
        value = jnp.take_along_axis(q, edges[:, None], 1)[:, 0]
        return dict(actions=actions, value=jnp.where(done, terminal_values, value), visits=visits,
            terminal=done, overflow=tree.embeddings[0].overflow.any(1), labels=labels,
            root_coverage=((visits > 0) & masks).sum(1) / jnp.maximum(masks.sum(1), 1))
    return search
