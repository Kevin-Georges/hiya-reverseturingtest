# AI tools used

> Kevin: edit this so it matches exactly what you did. Reviewers will ask about it.

## Tools
- **Claude (Anthropic) in the Claude app.** Idea selection, the first version of the code, tests and draft docs.
- **Claude Code (Anthropic, Claude Opus 5.5) in VS Code.** Later features, debugging on my machine, and updating the docs.
- **[Add anything else you used, e.g. Copilot, ChatGPT, or none.]**

## How I used them

**Choosing the problem.** I used Claude to research Hiya's products and public API docs (call screening,
deepfake detection, in-call whisper warnings, decoy bot) so I would not rebuild something Hiya already sells.
We went through several ideas and I picked the behavioural "reverse Turing test" because it complements
audio-based deepfake detection instead of competing with it.

**First version.** Claude generated the first version of the codebase from a design we agreed on: a simulated
full-duplex phone line, a VAD-based segmenter, the screener and its three probes, three scam-bot
variants, the scoring and the evaluation script. Claude also wrote fake backends (noise-burst "speech"
whose frames are fingerprinted so text still flows end to end), so the whole pipeline could be tested
without models.

**Later changes, made with Claude Code at my request:**
- An LLM red-team loop (`rtt/redteam.py`): an attacker LLM rewrites the scam bot's settings after every call.
  I then asked for the attacker to keep learning across runs (a memory file with its own lessons).
- A learned screener (`rtt/train.py`), and a mode where it retrains after every red-team round.
- After the screener flagged me as a bot on my own call, Claude Code found two bugs from the recording:
  a short noise during the greeting counted as my answer, and Bluetooth audio delay was not corrected for.
- At my request, the screener no longer interrupts normal callers: it varies its wording each call, waits for
  the caller to finish, and only runs the interruption and injection probes when the timing looks suspicious.
- An echo check (`python -m rtt.audio_io`), and a warning that keeps calls out of training when the mic
  picked up the screener's own voice. This came from my human calls, which turned out to have that problem.
- Rewriting the README, role card and write-up to match.

**What I did.** [Check and complete:]
- Set up Ollama, Whisper and audio devices on my machine, and ran the bot calls and red-team runs.
- Made human test calls, and reported what went wrong (flagged as a bot, being cut off mid-answer).
- Decided on the design changes above: learning attacker, learning defender, no interruptions for normal callers.
- [Anything else: code you reviewed or changed yourself, thresholds you chose, ...]

## Things I checked rather than trusted
- The scoring thresholds in `features.py` started as guesses. The numbers in the write-up come from my own
  real calls, not from the fake test runs.
- The simulated human (`simhuman.py`) is a test fixture only and is excluded from all results.
- My human calls were all affected by audio problems (see the write-up), so I report no false-alarm rate on people.
