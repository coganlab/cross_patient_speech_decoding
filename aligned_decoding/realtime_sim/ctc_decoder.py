"""
CTC Beam Search Decoder
*** Copied from https://gist.github.com/awni/56369a90d03953e370f3964c826ed4b0 ***

Author: Awni Hannun
This is an example CTC decoder written in Python. The code is
intended to be a simple example and is not designed to be
especially efficient.
The algorithm is a prefix beam search for a model trained
with the CTC loss function.
For more details checkout either of these references:
  https://distill.pub/2017/ctc/#inference
  https://arxiv.org/abs/1408.2873
"""

import numpy as np
import torch
import math
import collections

NEG_INF = -float("inf")

def make_new_beam():
  fn = lambda : (NEG_INF, NEG_INF)
  return collections.defaultdict(fn)

def logsumexp(*args):
  """
  Stable log sum exp.
  """
  if all(a == NEG_INF for a in args):
      return NEG_INF
  a_max = max(args)
  lsp = math.log(sum(math.exp(a - a_max)
                      for a in args))
  return a_max + lsp


def decode(probs, beam_size=100, blank=0):
  """
  Performs inference for the given output probabilities.
  Arguments:
      probs: The output probabilities (e.g. post-softmax) for each
        time step. Should be an array of shape (time x output dim).
      beam_size (int): Size of the beam to use during inference.
      blank (int): Index of the CTC blank label.
  Returns the output label sequence and the corresponding negative
  log-likelihood estimated by the decoder.
  """
  T, S = probs.shape
  probs = np.log(probs)

  # Elements in the beam are (prefix, (p_blank, p_no_blank))
  # Initialize the beam with the empty sequence, a probability of
  # 1 for ending in blank and zero for ending in non-blank
  # (in log space).
  beam = [(tuple(), (0.0, NEG_INF))]

  for t in range(T): # Loop over time

    # A default dictionary to store the next step candidates.
    next_beam = make_new_beam()

    for s in range(S): # Loop over vocab
      p = probs[t, s]

      # The variables p_b and p_nb are respectively the
      # probabilities for the prefix given that it ends in a
      # blank and does not end in a blank at this time step.
      for prefix, (p_b, p_nb) in beam: # Loop over beam

        # If we propose a blank the prefix doesn't change.
        # Only the probability of ending in blank gets updated.
        if s == blank:
          n_p_b, n_p_nb = next_beam[prefix]
          n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
          next_beam[prefix] = (n_p_b, n_p_nb)
          continue

        # Extend the prefix by the new character s and add it to
        # the beam. Only the probability of not ending in blank
        # gets updated.
        end_t = prefix[-1] if prefix else None
        n_prefix = prefix + (s,)
        n_p_b, n_p_nb = next_beam[n_prefix]
        if s != end_t:
          n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
        else:
          # We don't include the previous probability of not ending
          # in blank (p_nb) if s is repeated at the end. The CTC
          # algorithm merges characters not separated by a blank.
          n_p_nb = logsumexp(n_p_nb, p_b + p)
          
        # *NB* this would be a good place to include an LM score.
        next_beam[n_prefix] = (n_p_b, n_p_nb)

        # If s is repeated at the end we also update the unchanged
        # prefix. This is the merging case.
        if s == end_t:
          n_p_b, n_p_nb = next_beam[prefix]
          n_p_nb = logsumexp(n_p_nb, p_nb + p)
          next_beam[prefix] = (n_p_b, n_p_nb)

    # Sort and trim the beam before moving on to the
    # next time-step.
    beam = sorted(next_beam.items(),
            key=lambda x : logsumexp(*x[1]),
            reverse=True)
    beam = beam[:beam_size]

  best = beam[0]
  return best[0], -logsumexp(*best[1])


def beam_decode_torch(probs, beam_size=100, blank=0):
    """
    CTC beam search in PyTorch (GPU-compatible)

    probs: (T, S) log probabilities (after log_softmax)
    beam_size: beam width
    blank: index of CTC blank
    """
    T, S = probs.shape
    device = probs.device

    # Initialize the beam with empty prefix
    beam = {(): torch.tensor([0.0, NEG_INF], device=device)}  # [p_blank, p_nonblank]

    for t in range(T):
        next_beam = {}

        for s in range(S):
          p = probs[t, s]

          for prefix, (p_b, p_nb) in beam.items():
            if s == blank:
              n_p_b, n_p_nb = next_beam.get(prefix, torch.tensor([NEG_INF, NEG_INF], device=device))
              n_p_b = torch.logsumexp(torch.tensor([n_p_b, p_b + p, p_nb + p], device=device), dim=0)
              next_beam[prefix] = torch.tensor([n_p_b, n_p_nb], device=device)
              continue

            end_t = prefix[-1] if prefix else None
            n_prefix = prefix + (s,)
            n_p_b, n_p_nb = next_beam.get(n_prefix, torch.tensor([NEG_INF, NEG_INF], device=device))
            if s != end_t:
              n_p_nb = torch.logsumexp(torch.tensor([n_p_nb, p_b + p, p_nb + p], device=device), dim=0)
            else:
              n_p_nb = torch.logsumexp(torch.tensor([n_p_nb, p_b + p], device=device), dim=0)

            next_beam[n_prefix] = torch.tensor([n_p_b, n_p_nb], device=device)

            if s == end_t:
              n_p_b, n_p_nb = next_beam.get(prefix, torch.tensor([NEG_INF, NEG_INF], device=device))
              n_p_nb = torch.logsumexp(torch.tensor([n_p_nb, p_nb + p], device=device), dim=0)
              next_beam[prefix] = torch.tensor([n_p_b, n_p_nb], device=device)

        # Sort and trim the beam
        beam_items = list(next_beam.items())
        beam_items.sort(key=lambda x: torch.logsumexp(x[1], dim=0).item(), reverse=True)
        beam = dict(beam_items[:beam_size])

    best_prefix = max(beam.items(), key=lambda x: torch.logsumexp(x[1], dim=0).item())
    return best_prefix[0], -torch.logsumexp(best_prefix[1], dim=0).item()


def greedy_decode_batch(log_probs, blank=0):
    """
    log_probs: (B, T, C)
    returns: list of 1D LongTensors
    """
    best_paths = log_probs.argmax(dim=2)  # (B, T)

    decoded = []
    for b in range(best_paths.size(0)):
        path = best_paths[b]

        not_repeat = torch.ones_like(path, dtype=torch.bool)
        not_repeat[1:] = path[1:] != path[:-1]
        not_blank = path != blank

        decoded.append(path[not_repeat & not_blank])

    return decoded
