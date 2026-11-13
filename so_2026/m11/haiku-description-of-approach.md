
https://stackoverflow.com/beta/challenges/80015893/challenge-25-count-your-syllables

> Please include ... a brief description of your approach

I found it convenient to do some impedance matching between
the raw JSON input and a Word @dataclass.
Rather than stick with "stringly typed", I made
a PartOfSpeech Enum part of that.
In the end we wind up with a readily accessible
readonly vector of 200 Words.

There's also some state to carry across poems,
to satisfy the stated requirements.

- `self.seen` is a `set` that prevents a given word from being reused.
- To support the need for "diverse starter word syllable counts", we maintain `self.desired` number of syllables, and a cyclic index into it.

Most of the work is done by `_generate_line()` along with the
other helpers it calls. There's a fourfold part-of-speech requirement
across the entire poem, and the hack I chose was to focus on
adj + noun for "short" lines, and on adv + verb for "long" lines.

With that machinery in place, generating one or five poems
is just a matter of a little formatting.
