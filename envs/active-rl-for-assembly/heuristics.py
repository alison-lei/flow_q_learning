import pickle
import random
import numpy as np
import collections
from scipy.stats import linregress


class RobotGatedHeuristic:
    def __init__(self):
        self.expert_acts = False

    def reset(self):
        self.expert_acts = False

    def evaluate_state(self):
        if self.expert_acts and not self._is_uncertain():
            self.expert_acts = False
            # print("switch to:", self.expert_acts)
        elif not self.expert_acts and self._is_uncertain():
            self.expert_acts = True
            # print("switch to:", self.expert_acts)
        # else:
            # print("stay")
        # print()

    def _is_uncertain(self):
        raise NotImplementedError("_is_uncertain has not been defined for this heuristic type")


class RandomRobotGatedHeuristic(RobotGatedHeuristic):
    def __init__(self, config, intervention_seed):
        super().__init__()
        self.prob_from_robot_to_expert = config.prob_from_robot_to_expert
        self.prob_from_expert_to_robot = config.prob_from_expert_to_robot

        self.rng = np.random.default_rng(intervention_seed)
        print("INTERVENTION SEED IS ", intervention_seed)

    def _is_uncertain(self):
        if self.expert_acts:
            # if random < prob to switch to robot,
            # we want to switch to the robot,
            # we are certain (not uncertain)
            value = self.rng.random()
            # print(self.expert_acts, value, self.prob_from_expert_to_robot)
            return not (value < self.prob_from_expert_to_robot)
        else:
            # if random < prob to switch to expert,
            # we want to switch to the expert,
            # we are uncertain
            value = self.rng.random()
            # print(self.expert_acts, value, self.prob_from_robot_to_expert)
            return value < self.prob_from_robot_to_expert

class RandomRobotGatedHeuristicGaussianLength(RobotGatedHeuristic):
    """Random-trigger intervention with Gaussian-sampled length.

    Trigger (robot -> expert): each robot step, switch with fixed
    probability ``config.prob_from_robot_to_expert`` (same as the base
    ``RandomRobotGatedHeuristic``).

    Release (expert -> robot): on each robot -> expert transition, an
    intervention length is sampled from
    ``N(config.intervention_length_mean, config.intervention_length_std)``,
    clamped to ``>= 1`` and rounded to int. The expert is then released
    after exactly that many steps.

    Differs from ``RandomRobotGatedHeuristic`` (which samples Bernoulli
    each step to release, giving a geometric length distribution): here
    the length is drawn once per intervention.
    """

    def __init__(self, config, intervention_seed):
        super().__init__()
        self.prob_from_robot_to_expert = config.prob_from_robot_to_expert
        self.intervention_length_mean = config.intervention_length_mean
        self.intervention_length_std = config.intervention_length_std

        self.rng = np.random.default_rng(intervention_seed)
        self.intervention_length = 0
        self.current_intervention_steps = 0
        print("INTERVENTION SEED IS ", intervention_seed)

    def _is_uncertain(self):
        if self.expert_acts:
            return self.current_intervention_steps < self.intervention_length
        return self.rng.random() < self.prob_from_robot_to_expert

    def evaluate_state(self):
        was_expert = self.expert_acts
        super().evaluate_state()
        if not was_expert and self.expert_acts:
            length = self.rng.normal(
                self.intervention_length_mean,
                self.intervention_length_std,
            )
            self.intervention_length = max(1, int(round(length)))
            self.current_intervention_steps = 1
        elif was_expert and self.expert_acts:
            self.current_intervention_steps += 1

    def reset(self):
        super().reset()
        self.intervention_length = 0
        self.current_intervention_steps = 0


class RandomRobotGatedHeuristicEmpirical(RobotGatedHeuristic):
    """Random-trigger intervention with empirical span-length distribution.

    Both the trigger rate and the span-length distribution are loaded from
    a stats pickle (typically derived from a paired q_slope reference run)
    and stay fixed for the entire training run -- there is no curriculum
    component, so this is the matched-budget / matched-length-distribution
    baseline against q_slope.

    Pickle payload (dict):
        {
            "lambda": 0.025,
            "span_lengths": np.array([10, 11, 12, ..., 26], dtype=int),
        }

    ``lambda`` is the per-step Bernoulli rate sampled while the robot is in
    control (i.e., the firing probability over non-intervention steps).
    ``span_lengths`` is the empirical sample of intervention durations from
    the reference run; on each robot -> expert transition a length is drawn
    uniformly from this array with replacement.
    """

    def __init__(self, stats_path, intervention_seed):
        super().__init__()
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        self.lambda_ = float(stats["lambda"])
        self.span_lengths = np.asarray(stats["span_lengths"], dtype=int)
        if self.span_lengths.size == 0:
            raise ValueError(f"Empty span_lengths in {stats_path}")

        self.rng = np.random.default_rng(intervention_seed)
        self.intervention_length = 0
        self.current_intervention_steps = 0
        print("INTERVENTION SEED IS ", intervention_seed)
        print(
            f"Loaded empirical stats from {stats_path}: "
            f"lambda={self.lambda_}, n_span_samples={self.span_lengths.size}"
        )

    def _is_uncertain(self):
        if self.expert_acts:
            return self.current_intervention_steps < self.intervention_length
        return self.rng.random() < self.lambda_

    def evaluate_state(self):
        was_expert = self.expert_acts
        super().evaluate_state()
        if not was_expert and self.expert_acts:
            self.intervention_length = max(1, int(self.rng.choice(self.span_lengths)))
            self.current_intervention_steps = 1
        elif was_expert and self.expert_acts:
            self.current_intervention_steps += 1

    def reset(self):
        super().reset()
        self.intervention_length = 0
        self.current_intervention_steps = 0


class RandomRobotGatedHeuristicEmpiricalCurriculum(RobotGatedHeuristic):
    """Curriculum-matched random-trigger intervention.

    Same trigger / length mechanism as ``RandomRobotGatedHeuristicEmpirical``
    but the (lambda, span_lengths) pair changes every ``chunk_size`` env
    steps so the temporal allocation of interventions tracks the reference
    q_slope run. This is the baseline that controls for q_slope's tendency
    to intervene more at the start of training; differences against
    ``RandomRobotGatedHeuristicEmpirical`` isolate the curriculum effect,
    differences against q_slope itself isolate the within-episode timing
    effect.

    Pickle payload (dict):
        {
            "chunk_size": 500,
            "chunks": [
                {"lambda": 0.05, "span_lengths": np.array([...])},
                {"lambda": 0.04, "span_lengths": np.array([...])},
                ...
            ],
        }

    The env-step counter is incremented on every ``evaluate_state`` call
    and persists across episode resets so chunk boundaries align with
    cumulative training progress, not within-episode progress. Once the
    counter passes the last chunk, the final chunk's stats are reused for
    the rest of training.
    """

    def __init__(self, stats_path, intervention_seed):
        super().__init__()
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        self.chunk_size = int(stats["chunk_size"])
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive in {stats_path}")
        self.chunks = []
        for i, c in enumerate(stats["chunks"]):
            span_lengths = np.asarray(c["span_lengths"], dtype=int)
            lam = float(c["lambda"])
            # Empty span_lengths is allowed iff lambda == 0 (q_slope didn't
            # intervene in this chunk -> random shouldn't either, and the
            # sampling code path is never reached). Empty with lambda > 0
            # would be inconsistent input (positive trigger rate, no lengths
            # to draw).
            if span_lengths.size == 0 and lam > 0:
                raise ValueError(
                    f"Chunk {i} of {stats_path} has lambda={lam} > 0 but empty "
                    "span_lengths; cannot sample lengths when triggers fire."
                )
            self.chunks.append({"lambda": lam, "span_lengths": span_lengths})
        if not self.chunks:
            raise ValueError(f"No chunks in {stats_path}")

        self.rng = np.random.default_rng(intervention_seed)
        self.env_step = 0
        self.intervention_length = 0
        self.current_intervention_steps = 0
        print("INTERVENTION SEED IS ", intervention_seed)
        print(
            f"Loaded curriculum stats from {stats_path}: "
            f"chunk_size={self.chunk_size}, n_chunks={len(self.chunks)}"
        )

    def _current_chunk(self):
        idx = min(self.env_step // self.chunk_size, len(self.chunks) - 1)
        return self.chunks[idx]

    def _is_uncertain(self):
        if self.expert_acts:
            return self.current_intervention_steps < self.intervention_length
        return self.rng.random() < self._current_chunk()["lambda"]

    def evaluate_state(self):
        was_expert = self.expert_acts
        super().evaluate_state()
        if not was_expert and self.expert_acts:
            span_lengths = self._current_chunk()["span_lengths"]
            self.intervention_length = max(1, int(self.rng.choice(span_lengths)))
            self.current_intervention_steps = 1
        elif was_expert and self.expert_acts:
            self.current_intervention_steps += 1
        self.env_step += 1

    def reset(self):
        super().reset()
        self.intervention_length = 0
        self.current_intervention_steps = 0
        # env_step is NOT reset -- curriculum tracks cumulative training.


class RobotGatedHeuristicQValueSlopeFinishAfterStart(RobotGatedHeuristic):
    def __init__(self, config, window_size=20, threshold=0):
        super().__init__()
        self.q_values = collections.deque(maxlen=window_size)
        self.threshold = threshold
        self.window_size = window_size
        self.x = np.arange(self.window_size)
    
    def add_q_value(self, q_value):
        q_value = np.mean(q_value)
        self.q_values.append(q_value)

    def _is_uncertain(self):
        if self.expert_acts:
            return True
        else:
            if len(self.q_values) < self.window_size:
                return False
            slope, _, _, _, _ = linregress(self.x, self.q_values)
            return slope <= self.threshold
        
    def reset(self):
        super().reset()
        self.q_values.clear()


class RobotGatedHeuristicQValueSlopeRecover(RobotGatedHeuristic):
    """Slope-triggered intervention with Q-value-recovery exit.

    Trigger (robot -> expert): same as the FinishAfterStart variant -- the
    slope of the rolling Q-value window drops at/below ``threshold``.

    Release (expert -> robot): the most recent Q-value strictly exceeds
    the episode-wide maximum Q seen so far AND the slope of the rolling
    Q-value window is strictly greater than ``threshold`` (the same
    threshold used by the trigger -- symmetric to ``slope <= threshold``
    on the way in) AND the current intervention has lasted at least
    ``min_intervention_steps`` steps. The window must be full for the
    slope check to be evaluated; otherwise the expert is held.

    ``episode_max_q`` is tracked across **every** step regardless of who
    is acting (robot OR expert). Q-values produced during an intervention
    therefore raise the recovery bar, and the expert is only released
    once it strictly exceeds the running max.

    A per-episode cap (``max_interventions_per_episode``) bounds the number
    of distinct intervention segments; once the cap is hit, no further
    interventions are triggered for the rest of the episode (current
    intervention, if any, still terminates normally via the recovery rule).

    ``clear_q_values_on_intervention_end`` controls how the slope-window
    deque evolves across an expert segment:
      - False (default): Q-values are appended on every step, including
        during expert control, and the deque is preserved when control
        returns to the robot. The next slope check after release sees a
        window that includes expert-phase Q-values.
      - True: the deque is cleared on the expert -> robot transition, so
        the slope check resumes from a clean window. The robot then waits
        until ``window_size`` fresh Q-values have accumulated before the
        slope-based trigger can fire again.
    """

    def __init__(self, config):
        super().__init__()
        self.q_values = collections.deque(maxlen=config.window_size)

        self.threshold = config.threshold
        self.window_size = config.window_size
        self.x = np.arange(self.window_size)
        self.min_intervention_steps = config.min_intervention_steps
        self.max_interventions_per_episode = config.max_interventions_per_episode
        self.clear_q_values_on_intervention_end = config.clear_q_values_on_intervention_end

        self.episode_max_q = -np.inf
        self.last_q_value = None
        self.q_value_recovered_this_step = False
        self.current_intervention_steps = 0
        self.intervention_count = 0

        self._x_centered = (self.x - self.x.mean()).astype(np.float64)
        self._x_var_sum = float((self._x_centered ** 2).sum())
        # For window_size=20: _x_var_sum = 665.0


        print(f"Initializing QValueSlopeRecover with window size {self.window_size}")
    
    def _slope(self):
        y = np.fromiter(self.q_values, dtype=np.float64, count=self.window_size)
        return float(self._x_centered @ y) / self._x_var_sum

    def add_q_value(self, q_value):
        # q_value = float(np.mean(q_value))
        self.q_values.append(q_value)
        self.last_q_value = q_value
        # Track the all-episode max here so _is_uncertain can stay pure.
        self.q_value_recovered_this_step = q_value > self.episode_max_q
        if self.q_value_recovered_this_step:
            self.episode_max_q = q_value

    def _is_uncertain(self):
        if self.expert_acts:
            # Hold the expert in control until all gates clear:
            #   1. minimum intervention duration met,
            #   2. last Q-value strictly exceeds the episode max,
            #   3. slope of the rolling Q-value window > threshold
            #      (symmetric to the trigger's slope <= threshold).
            if self.current_intervention_steps < self.min_intervention_steps:
                return True
            if not self.q_value_recovered_this_step:
                return True
            # print("COMPARE SLOPE CALCULATED:", self._slope(), linregress(self.x, self.q_values)[0])
            return self._slope() <= self.threshold
            # slope, _, _, _, _ = linregress(self.x, self.q_values)
            # return slope <= self.threshold

        # Robot in control -- only trigger if we haven't exhausted the
        # per-episode intervention budget and the slope says drop-off.
        if (self.max_interventions_per_episode is not None
                and self.intervention_count >= self.max_interventions_per_episode):
            return False
        if len(self.q_values) < self.window_size:
            return False
        # print("COMPARE SLOPE CALCULATED:", self._slope(), linregress(self.x, self.q_values)[0])
        return self._slope() <= self.threshold
        # slope, _, _, _, _ = linregress(self.x, self.q_values)
        # return slope <= self.threshold

    def evaluate_state(self):
        was_expert = self.expert_acts
        super().evaluate_state()
        if not was_expert and self.expert_acts:
            # Fresh expert segment: start the per-segment step counter at 1
            # and bump the intervention tally.
            self.current_intervention_steps = 1
            self.intervention_count += 1
        elif was_expert and self.expert_acts:
            # Continuing expert segment: count this step.
            self.current_intervention_steps += 1
        elif was_expert and not self.expert_acts and self.clear_q_values_on_intervention_end:
            # Expert -> robot handover: optionally drop the slope window so
            # the next trigger evaluates only post-intervention Q-values.
            self.q_values.clear()

    def reset(self):
        super().reset()
        self.q_values.clear()
        self.episode_max_q = -np.inf
        self.last_q_value = None
        self.q_value_recovered_this_step = False
        self.current_intervention_steps = 0
        self.intervention_count = 0


# This class is created to collect data for analysis
# We have human-gated intervention but with a condition:
# If a human intervention has started, it has to continue
# till the end of the episode
class HumanGatedHeuristicFinishAfterStart(RobotGatedHeuristic):
    def _is_uncertain(self):
        return self.expert_acts

    def evaluate_state(self):
        pass

    def start_human_intervention(self):
        # print("START HUMAN CONTROL")
        self.expert_acts = True


# Human gates BOTH the start and end of the intervention via keypresses.
# evaluate_state() polls the viewer keyboard and TOGGLES expert_acts on a
# fresh press. start_/end_human_intervention() are kept for external
# triggers (e.g., KeyboardToggleHG-based test harnesses).
class HumanGatedHeuristicWithButtonPresses(RobotGatedHeuristic):
    def __init__(self, env, toggle_keys=None):
        super().__init__()
        # Lazy import: pulling KeyboardExpert at module load would force
        # robosuite/mujoco imports even for callers that don't need HG.
        from sim_env.envs.modified_wrappers import KeyboardExpert
        self.keyboard = KeyboardExpert(env)
        self.toggle_keys = toggle_keys if toggle_keys is not None else [32]
        self._last_consumed_ts = 0.0

    def _is_uncertain(self):
        return self.expert_acts

    def evaluate_state(self):
        """Poll keyboard; toggle expert_acts on a fresh press.

        Returns True iff a press was consumed (state flipped) this call.
        """
        self.keyboard._register_viewer_callbacks_if_needed()
        latest_ts = -1.0
        for code in self.toggle_keys:
            ts = self.keyboard._last_key_event.get(int(code), -1e9)
            if ts > latest_ts:
                latest_ts = ts
        if latest_ts > self._last_consumed_ts:
            self._last_consumed_ts = latest_ts
            self.expert_acts = not self.expert_acts
            return True
        return False

    def start_human_intervention(self):
        self.expert_acts = True

    def end_human_intervention(self):
        self.expert_acts = False

