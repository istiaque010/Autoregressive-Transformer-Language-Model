# Evaluation — baseline

- Checkpoint: `checkpoints/baseline/model_best.pt` (step 5000)
- Parameters: 5,322,240 total (3,159,552 non-embedding), 20.3 MB fp32
- **Test** loss **2.2700**, perplexity **9.68** (validation: loss 2.2705, perplexity 9.68)
- Device: cpu | torch 2.14.1+cpu | precision fp32 | peak memory of the evaluation process 781 MB

## Sampling strategies

Up to 512 new tokens per story. Distinct-n uses the first 100 words of each story, in groups of one story per prompt (mean ± std across groups).

| Setting | Stories | Distinct-1 | Distinct-2 | Rep-4 rate | Flagged repetitive | EOS rate | Hit token limit | Avg tokens |
|---|---|---|---|---|---|---|---|---|
| *real stories* | 10,995 | 0.331 ± 0.015 | 0.764 ± 0.026 | 0.016 | 5% by definition | — | — | — |
| greedy | 8 | 0.151 ± 0.000 | 0.391 ± 0.000 | 0.488 | 100% | 75% | 25% | 268 |
| temp_0.7 | 40 | 0.263 ± 0.008 | 0.655 ± 0.010 | 0.051 | 30% | 88% | 12% | 201 |
| temp_1.0 | 40 | 0.358 ± 0.019 | 0.799 ± 0.021 | 0.006 | 0% | 98% | 2% | 209 |
| top_k_20 | 40 | 0.297 ± 0.022 | 0.739 ± 0.037 | 0.018 | 5% | 90% | 10% | 245 |
| top_k_50 | 40 | 0.322 ± 0.007 | 0.768 ± 0.016 | 0.016 | 2% | 88% | 12% | 236 |
| top_p_0.90 | 40 | 0.310 ± 0.013 | 0.738 ± 0.017 | 0.016 | 2% | 92% | 8% | 214 |
| top_p_0.95 | 40 | 0.336 ± 0.006 | 0.782 ± 0.014 | 0.013 | 0% | 95% | 5% | 213 |

A story is flagged repetitive when its Rep-4 rate is above 0.057, the 95th percentile of real validation stories.

## Generation speed

Fixed length: 200 tokens, 3 runs, 24 threads.

| Setting | Tokens/s | ms per token | Seconds for a 200-token story |
|---|---|---|---|
| greedy | 365.8 | 2.73 | 0.55 |
| temp_0.7 | 332.3 | 3.01 | 0.60 |
| temp_1.0 | 344.1 | 2.91 | 0.58 |
| top_k_20 | 345.9 | 2.89 | 0.58 |
| top_k_50 | 340.5 | 2.94 | 0.59 |
| top_p_0.90 | 278.9 | 3.59 | 0.72 |
| top_p_0.95 | 266.1 | 3.76 | 0.75 |

## Training

- Hardware: Intel64 Family 6 Model 198 Stepping 2, GenuineIntel, 24 logical CPUs, 63.7 GB RAM, 24 threads, CPU only
- Precision fp32 | batch 8 sequences × 4 accumulation steps = effective batch 32 sequences (8,192 tokens)
- 5,000 steps, 41M tokens seen, data `data`
- Training time **1.0 h**, average 11,092 tokens/s, peak memory 989 MB (highest value in `log.csv`)

## Error analysis (60 stories, top_p_0.90)

- repetition: 2/60 (3%, auto-flagged)
- incomplete: 4/60 (7%, auto-flagged)
- hit_token_limit: 4/60 (7%, auto-flagged)
- Label contradiction / grammar / entity consistency by hand in `results\generations\baseline\error_analysis.csv`.

## Example generations

**greedy**

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary monster. The monster was scared and didn't know what to do. Lily's mom told her that it was a scary monster. Lily was scared and didn't know what to do. She decided to go to the monster and see if it was scary. The monster was scared and didn't know what to do. Lily's mom told her that it was a scary monster. The monster was scared and didn't know what to do. Lily's mom told her that it was a scary monster. The monster was scared and didn't know what to do. Lily learned that it's important to be brave and listen to her mom. From that day on, Lily always remembered to be careful when she was scared.

**temp_0.7**

> Once upon a time there was a girl named Lily. She had a cozy bed that she loved to sleep. One day, she went to bed and asked her mom what she could do. Her mom said, "Don't worry, I'll help you read the book." Lily understood and went to night, dreaming of herself. She dreamed of the book and the library. She hoped to find a new book to read on the book. But when she tried, she tried to read the book with her mom. She thought it was very creative. Then, Lily's mom said, "Lily, we need to read the book together." Lily was happy to read more books. When they finished, she was very happy. She said, "Thank you, mom. I love you." Her mom said, "I love you too, Lily, but you always remember to be kind to others. You are a good friend."

**temp_1.0**

> Once upon a time, there was a little girl named Lucy. Lucy was very good at once with three walls. One day Lucy liked to imagine. She wanted to explore her newcess and see what was an independent little lives. Suddenly, a big appeared behind a headed towards them. When Lucy reached the gate, she realised she was careless. Her eyes wouldn't come out. Lucy heard coming from hand and held onto the zip. Startled, Lucy stepped, the pin began right forward. Shely approached it tightly. It seemed so far away. She was very proud to see the magical light, so she asked her parents more books about her holiday. Her parents laughed and said, "Come on, let's turn to the top of the slide!"

**top_k_20**

> Once upon a time there was a very special girl named Emily. Emily was three years old and loved to play in the sunshine. One day, Emily wanted to make some special friends to make them happy. She put on her apron and ran outside to get a new one. She was so excited to try the big orange apple. She quickly jumped the apple up and down. Suddenly, as Emma heard a voice. It said, â€œWhat are you having, Emily?â€ Emily looked down and saw a kind bear. She was very excited to say hello back. â€œI like this!â€ she asked the bear. â€œIâ€™ll try.â€ Emily asked sadly. â€œYes, you can!â€ Emily answered. â€œItâ€™s too hard work!â€ The bear was so happy that Emily gave the bear a big hug. â€œThere is a magical surprise!â€ Emily smiled. â€œCan I touch it?" âYes, please!â€ The bear replied. â€œYes, so, you can touch it.â€ They continued to look and looked, until they were all gone. At the end of them were all safe and cozy. Lily's mother said â€œOh "I'm okay, why don't you go there?â€ Emily replied. â€œI told my mother. â€œWhy donâ€™t you explore the tree?â€ Her mother explained that he had to be very careful and listen to her. â€œThis is for the birds to be careful.â€ She told Emma that they would never go too far away.

**top_k_50**

> Once upon a time, there was a little boy who liked to skip. He would skip for all the time. One sunny day, he saw a big hill with a long branches. He stepped inside and started to ride his feet. He went up, down, eager to see how heavy the hill was there. He followed the hill and they went up the hill. He had so much fun that he started his speed as fast as he could. He zoomed back higher and higher and higher, so they could see the clouds in the sky. His little dad were so proud of him. He was very proud of himself for running so fast that he ran up to the hill again. The little boy had been so pleased to be careful and be careful with his dad. He knew that that the hill was cool and the view wasn't always there to explore!

**top_p_0.90**

> Once upon a time there was a lazy cat named Tim. Tim loved to play with his friends, but sometimes he accidentally broke the tree and it went to the park. Tim was sad because he loved his family very much. But then, he saw something that made a big, mean owl. The owl asked, "Why can you make your nest so I can't match it." Tim said, "I can design it for my family." The owl used its board to make a gentle nest for Tim to help. The owl showed the wise owl a match from his collection. He decided to help the owl and strong seeds. Tim worked hard and soon the owl was fixed, and he was very happy. From that day on, Tim made sure to build things and make everything really heavy. The end.

**top_p_0.95**

> Once upon a time, there was a soft little bird named support. One day, a small bird was flying in a tree with a long tail. The bird wanted to see the world go. So, the bird flew up and waved at the others. The bird flew and flew high in the sky. Suddenly, the wind slipped and fell into the air. The bird was stuck and couldn't catch up. The big bird could not move with the small bird. The bird felt sad and sorry for the pretty bird. It wanted the bird to help a bird. It had an idea. The kind bird looked at the bird and said, "No, you can't help the little bird. You need to find something that will help you." The little bird walked to its room and gently found the little bird's favorite toy. The bird loved the bird, and thanked the bird for helping. From that day on, the bird became good friends and played together every day.
