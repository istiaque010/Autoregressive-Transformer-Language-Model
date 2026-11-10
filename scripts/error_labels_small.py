"""Phase 13: hand labels for results/generations/small/error_analysis.csv (60 stories, top-p 0.90).

Labelled on 2026-11-06 by reading every story. Rules used for each label:
- contradiction: something clashes with an earlier fact in the story or doesn't follow from it
  (a place, an object, an event, a feeling).
- grammar: at least one broken sentence (wrong word form, missing words, invented word).
- entity_consistency: a character changes name, gender or species, appears without being introduced
  in a way that confuses who is who, or the wrong character speaks or acts.
Notes also say whether the automatic repetition flag was a real loop ("rep: real"), a false alarm,
or missed a loop ("rep: missed").

Kept as a script, not only in the CSV, so the labels survive if the CSV is regenerated.
Usage: python scripts/error_labels_small.py   (writes the labels into the CSV and prints the counts)
"""
import csv

PATH = "results/generations/small/error_analysis.csv"
HAND = ["contradiction", "grammar", "entity_consistency"]

# id: (contradiction, grammar, entity_consistency, notes)
LABELS = {
    0: (0, 0, 0, "mostly coherent; 'very surprised, and she was very surprised'; ends 'tired from all the laughing' (no laughing)"),
    1: (1, 0, 1, "'Lily was just a toy'; 'Max and Lily played with Max'"),
    2: (1, 1, 0, "'kicked it with his hands'; 'his ball hit the ball'; 'He liked to be park tomorrow'"),
    3: (1, 0, 0, "mouse says 'Let me help you' then 'wouldn't help'; cat 'never woke up'; rep: false alarm (short story)"),
    4: (0, 1, 0, "no character at start (prompt); 'The bear's happiness was always it never seemed too good'"),
    5: (1, 0, 1, "park then 'Mom was busy in the kitchen'; Ben says 'Ben, can we go, Mom?'; rep: real loop ('He liked his mom.' x3)"),
    6: (0, 0, 0, "clean"),
    7: (1, 1, 1, "'twist the key on the key', invented 'keyhopper'; 'angry and very happy'; mom/Sara 'she' mixed; rep: real"),
    8: (1, 0, 0, "'You already have enough money'; 'sad to leave the store' while at home"),
    9: (1, 0, 0, "knife 'snapped on Lily's foot'; new knife 'put on Lily's foot'"),
    10: (1, 1, 1, "who owns which ball keeps flipping ('lend my ball to me'); 'give her his ball to Sue'"),
    11: (1, 0, 0, "cat says 'Yes, you can eat me!' then is upset; moral does not follow"),
    12: (0, 0, 0, "no character at start (prompt); ending 'promised to stay friends' with no one"),
    13: (1, 0, 1, "dad appears from nowhere; parents 'came down the tree'; 'You did not climb the tree' (he did); rep: real but mild"),
    14: (1, 1, 1, "dragon 'chirps', child 'opened its beak' (child becomes a bird); 'looking for the child a family of bird'"),
    15: (1, 1, 1, "'the key was locked and the key was locked'; bear 'was his friend, the boy. He was a book'; rep: real + 'She wished' list"),
    16: (0, 1, 0, "'another new and adventure', 'use his task to climb'; moral does not follow"),
    17: (1, 0, 0, "toy 'getting dirty' then 'also clean'; 'got in the water and dried them'"),
    18: (1, 0, 1, "'a new boy... His name was Lily... she'; ball 'called broccoli'; Tom suddenly 'deaf'"),
    19: (1, 1, 1, "'The cat was happy to see the cat'; 'other things in the garden, too much food'"),
    20: (1, 1, 1, "'the brave one' is she then he; starts packing after arriving; 'She followed her lead and carefully'"),
    21: (1, 1, 1, "'My dog is gone!' (no dog); Ben says 'I love you, Ben. You are my dog'; 'brown apple... unique blue'"),
    22: (1, 1, 1, "person/creature/man/monster for one character; 'steal some of his own food'; 'could not have no food'"),
    23: (1, 0, 1, "found the key then go to the store to find it; 'his dad' for Sara's dad"),
    24: (1, 1, 0, "'a big grapes'; soap to feel better; grapes 'all green'"),
    25: (1, 0, 0, "she is talking to mom, then 'her mom came in'"),
    26: (1, 0, 1, "dad 'at work' and 'coming home' at once; whose ball flips; dad says 'I was a brave boy'; rep: real ('I'm sorry, dad.' x3); cut at 512"),
    27: (1, 0, 1, "'The dog asked the cat to eat the cat'; 'the dog ate the cat' then cat is happy"),
    28: (1, 0, 1, "boat 'ran away', then 'the pirate' (never introduced); house 'filled with golden treats'"),
    29: (1, 0, 1, "mom's line credited to Ben; 'Ben and Mia' (Mia never introduced); caught the balloon then 'You can't catch it'; rep: real"),
    30: (1, 1, 1, "dragon becomes 'The bear' at the end; 'keep my himself as a gift'"),
    31: (1, 1, 1, "Ben's hammer becomes Sara's; 'tried to pull it, but it did not'; dark ending; 'could not breathe' x2 (rep: missed)"),
    32: (1, 0, 0, "temple -> tent -> 'university where they go to a camp'; 'You're welcome' with no thanks"),
    33: (1, 1, 0, "wind 'washed the butterfly'; 'her mom's ow'; 'you should not shoot things'"),
    34: (1, 0, 1, "Lily appears from nowhere; 'She was in the tree... landed on his head'; 'wished he was still alive'; 'She wished' x3 (rep: missed)"),
    35: (1, 0, 1, "every animal is 'the cat', so you can't tell who is who; cat 'flew away'; rep: false alarm (caused by name confusion)"),
    36: (1, 0, 1, "'a little mouse named Tim. Tim was a very obedient cat'; 'picked up the cat'"),
    37: (1, 1, 1, "pandas in the park; 'an pandas', 'pandasplum', 'squ helicopter'; bird/Ben lines mixed; cut at 512"),
    38: (1, 0, 1, "bee becomes a butterfly; medal -> penny -> diamond -> crystal"),
    39: (1, 0, 1, "key -> lock -> necklace; mom says 'my key', 'have you as my coin'"),
    40: (0, 0, 0, "coherent; friends appear at the end"),
    41: (1, 0, 0, "'the bug was trying to catch it'; 'flew away with the bug'; happy that the bug didn't stop; 'could not eat the bug'"),
    42: (1, 0, 0, "'roll the ball down the tree'; dog 'get dirty'; rep: real ('the ball and the ball' x2)"),
    43: (1, 0, 0, "'cat ate it all up' (the bowl); 'not supposed to be food'; rep: real but mild"),
    44: (1, 0, 1, "'she'/Lily never introduced (prompt); giraffe in the garden; 'wouldn't get lost again'"),
    45: (1, 0, 1, "man 'looked like a real ball'; 'bad man' then 'shy'; 'Lily and Ben' (Lily new); rep: real loop of 'You are welcome... my friend'; cut at 512"),
    46: (1, 0, 0, "key -> penny -> coin; 'hide in a secret place' vs 'explore the world'"),
    47: (1, 1, 0, "'opened the key and saw many numbers'; 'maybe the can open it'; 'I have a brain!'"),
    48: (1, 1, 0, "'finished his eat'; dad trips, Timmy says sorry"),
    49: (1, 0, 1, "Lily says 'My name is Max'; Max 'barked back' (boy becomes dog); 'closed the lid' of nothing"),
    50: (1, 0, 1, "Sam appears from nowhere; 'He waved at Tom' (Tom waves at himself); 'jewels and jewels'; rep: real; cut at 512"),
    51: (1, 1, 0, "'The cat opened the branch'; 'even the bird was small and should let the cat eat his food'"),
    52: (1, 1, 1, "'climb the tree, instead of climbing it'; owl she -> he; 'dizzy after all the sudden'"),
    53: (1, 0, 1, "'Ben and Lily' (Lily new); 'the boy' for Ben; whose ball flips; bird 'had shiny teeth'; rep: real"),
    54: (0, 0, 0, "coherent ('pray together to make a cozy bed' is odd)"),
    55: (1, 0, 0, "key 'in her pocket' x4; 'put the key in the lock, so she could carry it inside'; box appears (rep: missed, mild)"),
    56: (1, 1, 0, "hungry -> 'took off her clothes'; 'didn't feel sick or so anymore'; 'played with them with her'"),
    57: (1, 1, 0, "favourite toy is 'a long neck'; 'big, but still it was her strong'"),
    58: (1, 0, 1, "Sam 'opened the box' and took out a ball; Sam offers 'my kite' (Tom's kite)"),
    59: (1, 0, 1, "cat 'ran back to the cat'; 'so full that it had finally found something tasty'"),
}


def is_yes(v):
    return v in ("yes", "True")


def main():
    with open(PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(LABELS) and {int(r["id"]) for r in rows} == set(LABELS)
    for r in rows:
        *flags, note = LABELS[int(r["id"])]
        for k, v in zip(HAND, flags):
            r[k] = "yes" if v else "no"
        r["notes"] = note
    with open(PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(f"{n} stories, {PATH}")
    for k in ["repetition", "incomplete", "hit_token_limit", *HAND]:
        c = sum(is_yes(r[k]) for r in rows)
        print(f"  {k:<20} {c:>2}/{n} = {c / n:.0%}")
    kinds = ["repetition", "incomplete", *HAND]
    clean = [r["id"] for r in rows if not any(is_yes(r[k]) for k in kinds)]
    print(f"  no error of any kind: {len(clean)}/{n} (ids {', '.join(clean)})")
    flagged = [r for r in rows if r["repetition"] == "True"]
    print(f"  repetition flag: {sum('rep: real' in r['notes'] for r in flagged)} real, "
          f"{sum('false alarm' in r['notes'] for r in flagged)} false alarms of {len(flagged)}; "
          f"missed loops: {sum('rep: missed' in r['notes'] for r in rows)}")
    errs = [sum(is_yes(r[k]) for k in HAND) for r in rows]
    print(f"  hand-labelled error types per story: 0: {errs.count(0)}, 1: {errs.count(1)}, 2: {errs.count(2)}, 3: {errs.count(3)}")


if __name__ == "__main__":
    main()
