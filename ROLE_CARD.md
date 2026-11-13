# Role card for human test calls

You will call the screener as a person. It takes about a minute.

## Before you start

1. Put your **headphones in your ears**, at a moderate volume. If the mic can hear the screener, the call is useless.
2. Check the audio (stay quiet while it plays):
   ```bash
   python -m rtt.audio_io
   ```
   It must print **OK**. If it prints ECHO, see the fixes it suggests.
3. Start the call:
   ```bash
   python -m rtt.run --caller human --speak-verdict
   ```
   With Bluetooth headphones (e.g. AirPods), add `--extra-latency-ms 200`.

The screen shows each line the screener says, and `heard you: started +0.4s ...` when it hears you.
A **negative** number on every answer, or a WARNING at the end, means the mic is hearing the screener:
fix the audio and record again.

## What to say

Glance at this, don't read it out word for word: reading changes your timing. Use a different role each call,
and change the details every time.

| The screener asks... | Scam caller (most calls) | Delivery driver | GP receptionist |
|---|---|---|---|
| who's calling | "Hi, it's Sam from the Northbank fraud team." | "Hiya, it's Jordan from Swiftline Deliveries." | "Hello, it's Priya from Oakfield Surgery." |
| what it's about | "There's a suspicious payment on your card." | "I've got a parcel but I can't find the house." | "Just confirming your appointment tomorrow." |
| what happened | Talk for 15 to 20 seconds: the £499 payment, the hold on it, needing the card number and a code. | Talk for 15 to 20 seconds: going up and down the road, missing numbers, the sat nav. | Talk for 15 to 20 seconds: the follow-up, arriving early, bringing medication. |
| a follow-up question | Answer however makes sense. | | |
| a callback number | "Ring the number on the back of your card." | "07700 900123." (a fictional number) | "07700 900456." |

If the screener does anything unexpected, react however you naturally would.

## Rules

- **Don't try to beat the detector.** React naturally. If you know what it checks for, you'll behave
  unnaturally around it; note that as a limitation.
- Record at least 5 clean calls before training the screener (`python -m rtt.train`), more if you can,
  spread over different sessions. Calls from more than one person are much more convincing.
