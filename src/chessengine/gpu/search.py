"""Deterministic uniform-prior PUCT using mctx's batched tree machinery."""
import jax
import jax.numpy as jnp
import mctx
from functools import lru_cache
from . import rules
from .network import apply


def evaluate(params, state, kind):
    return jax.lax.cond(kind == 0, lambda _: rules.handcrafted(state),
                        lambda _: apply(params, rules.features(state)), None)


def selection(_key, tree, index, _depth=0):
    visits = tree.children_visits[index]
    # mctx child values are STM at the child; discount negates exactly once.
    q = tree.children_rewards[index] + tree.children_discounts[index] * tree.children_values[index]
    q = jnp.where(visits > 0, q, 0)
    logits = tree.children_prior_logits[index]
    legal = jnp.isfinite(logits)
    priors = legal / jnp.maximum(legal.sum(), 1)
    parent_visits = visits.sum()
    score = q + 1.5 * priors * jnp.sqrt(jnp.maximum(parent_visits, 1)) / (1 + visits)
    return jnp.argmax(jnp.where(legal, score, -jnp.inf)).astype(jnp.int32)


@lru_cache(maxsize=8)
def make_search(simulations):
    if simulations < 1:
        raise ValueError('Positive fixed simulation budget required')

    def search(params, states, kinds):
        done, terminal_values = jax.vmap(rules.terminal)(states)
        values = jax.vmap(evaluate)(params, states, kinds)
        values = jnp.where(done, terminal_values, values)
        masks = states.game.legal_action_mask
        masks = jnp.where(done[:, None], jax.nn.one_hot(jnp.zeros(len(done), jnp.int32), 4672, dtype=bool), masks)
        root = mctx.RootFnOutput(prior_logits=jnp.where(masks, 0., -jnp.inf),
                                 value=values, embedding=(states, kinds))

        def recurrent(parameters, _key, actions, embeddings):
            before, evaluator_kinds = embeddings
            was_done, was_value = jax.vmap(rules.terminal)(before)
            after = jax.vmap(lambda s, a, d: jax.lax.cond(d, lambda _: s,
                             lambda _: rules.step(s, a), None))(before, actions, was_done)
            ended, terminal_value = jax.vmap(rules.terminal)(after)
            value = jax.vmap(evaluate)(parameters, after, evaluator_kinds)
            # Terminal transitions pay parent-perspective reward; no extra sign.
            reward = jnp.where(was_done, was_value, jnp.where(ended, -terminal_value, 0.))
            discount = jnp.where(ended | was_done, 0., -1.)
            value = jnp.where(ended, 0., value)
            mask = after.game.legal_action_mask
            # A dummy absorbing action keeps mctx finite on terminal leaves.
            mask = jnp.where(ended[:, None], jax.nn.one_hot(jnp.zeros(len(ended), jnp.int32), 4672, dtype=bool), mask)
            output = mctx.RecurrentFnOutput(reward=reward, discount=discount,
                prior_logits=jnp.where(mask, 0., -jnp.inf), value=value)
            return output, (after, evaluator_kinds)

        tree = mctx.search(params, jax.random.PRNGKey(0), root=root, recurrent_fn=recurrent,
            root_action_selection_fn=selection, interior_action_selection_fn=selection,
            num_simulations=simulations, invalid_actions=~masks)
        visits = tree.children_visits[:, 0]
        actions = jnp.argmax(visits, axis=1).astype(jnp.int32)
        q = tree.children_rewards[:, 0] + tree.children_discounts[:, 0] * tree.children_values[:, 0]
        chosen_value = jnp.take_along_axis(q, actions[:, None], axis=1).squeeze(1)
        return dict(actions=actions, value=jnp.where(done, terminal_values, chosen_value), visits=visits,
                    terminal=done, overflow=states.overflow,
                    root_coverage=((visits > 0) & masks).sum(1) / jnp.maximum(masks.sum(1), 1))

    return jax.jit(search)
