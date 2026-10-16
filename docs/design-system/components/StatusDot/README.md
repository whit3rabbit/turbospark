# StatusDot

The model status dot and its loader capsule: what model is selected and whether it is ready, working, or not loaded.

- 7pt dot. Ready and working use `accent`. Not loaded, none installed, and none selected use `ink-secondary`.
- Working (installing, opening) breathes with `TSMotion.breathe` between 45% and 100% opacity, only while the work is live. Static under Reduce Motion.
- The dot never carries meaning alone: the capsule always says "Ready", "Not loaded", or the stage in words.
- The model id and context size are verbatim, in `code`.
- The capsule offers the one next step: Load when not loaded, Eject when ready.

The consumer supplies the model state and the accent.
