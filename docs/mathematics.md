# Learning walkthrough

## 1. Values and decisions

The network receives a position and returns a scalar `v(s)` in [-1, 1], from the perspective of the player whose turn it is. Positive means favorable for that player. It is an advantage estimate, not automatically a calibrated win probability.

A parent and child position have opposite players to move. Therefore negamax uses `value(parent action) = -value(child)`. Missing or duplicating this minus sign makes an engine prefer its opponent's good positions. Forced terminal wins/losses have search scores outside the network range so an approximate evaluator cannot override a known mate.

Alpha-beta pruning avoids branches that cannot change a decision. Iterative deepening completes depths in order; if the budget runs out, the engine returns the last fully completed result, or a legal fallback if even depth one was interrupted. Quiescence extends tactical positions but is bounded to avoid unbounded capture/check sequences.

## 2. Supervised fitting

The teacher returns estimated win, draw, and loss counts. The label is:

`y = (wins - losses) / (wins + draws + losses)`

The loss `mean((v_theta(s) - y)^2)` is differentiable through the network; chess rules and search do not need gradients. AdamW updates the network weights. Training loss measures fit to seen examples; validation loss determines checkpoint selection and exposes overfitting. The final position test set stays separate.

Questions to understand at milestone one: Why does side-to-move perspective matter? Why split by game before sampling positions? Why can teacher MSE decrease while playing strength gets worse?

## 3. Monte Carlo reinforcement learning

In a completed game, the final result supplies the return. For every recorded state, `z` is +1 if its player eventually wins, -1 if they lose, and 0 for a draw. There are no intermediate rewards and no discounting (gamma=1).

A batch proceeds as follows:

1. Freeze the actor weights.
2. Generate games with search plus early-game exploration.
3. Observe completed outcomes and attach returns to states.
4. Sample equally from each completed game.
5. Minimize `mean((v_theta(s) - z)^2)`.
6. Use the updated evaluator to generate the next batch.

This is Monte Carlo policy evaluation embedded in approximate policy iteration: search chooses a policy from the evaluator, then outcomes teach the evaluator about that policy. It is reinforcement learning even though the regression update resembles supervised learning: the targets arise from interaction under a policy affected by the model.

We do not backpropagate through discrete moves, use PPO, or claim AlphaZero. Bootstrapped TD targets and learned policies would be separate extensions.

Exploration broadens experience but can degrade the quality of the behavior policy. Outcomes have high variance and are delayed. Mostly drawn self-play can pull values toward zero. Excluding truncated games avoids false rewards but biases the dataset toward games that finish. We measure these effects rather than assuming self-play helps.

Questions at milestone two: Why freeze the actor for a batch? Why not label truncated games as draws? Why is a decreasing RL loss not sufficient evidence? What changes when the policy that generates data changes?

## 4. Experimental inference

Match score is `(wins + 0.5 * draws) / games`. An opening pair consists of two games with colors reversed. Bootstrap whole pairs because the two games share an opening and are not independent.

An interval entirely above 50% supports improvement against that opponent at that budget. An interval spanning 50% is inconclusive, not proof of equality. A narrow interval from many openings does not capture variation across training seeds; report seeds separately.

Choose checkpoints using development games, then freeze them before final matches. Repeatedly selecting a model using the final test turns that test into development data. Fixed nodes compare playing decisions under equal search work; fixed time additionally includes inference efficiency.

Questions at milestone three: What exactly does the evidence support? What does it not support? Is a gain specific to one seed, opponent or search budget? Would the conclusion survive removing move-cap adjudications?
