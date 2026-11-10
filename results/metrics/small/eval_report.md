# Evaluation — small

- Checkpoint: `checkpoints/small/model_best.pt` (step 10000)
- Parameters: 13,891,584 total (10,647,552 non-embedding), 53.0 MB fp32
- **Test** loss **1.7473**, perplexity **5.74** (validation: loss 1.7467, perplexity 5.74)
- Device: cpu | torch 2.14.1+cpu | precision fp32 | peak memory of the evaluation process 941 MB

## Sampling strategies

Up to 512 new tokens per story. Distinct-n uses the first 100 words of each story, in groups of one story per prompt (mean ± std across groups).

| Setting | Stories | Distinct-1 | Distinct-2 | Rep-4 rate | Flagged repetitive | EOS rate | Hit token limit | Avg tokens |
|---|---|---|---|---|---|---|---|---|
| *real stories* | 10,995 | 0.331 ± 0.015 | 0.764 ± 0.026 | 0.016 | 5% by definition | — | — | — |
| greedy | 8 | 0.237 ± 0.000 | 0.586 ± 0.000 | 0.162 | 75% | 88% | 12% | 262 |
| temp_0.7 | 40 | 0.275 ± 0.008 | 0.693 ± 0.014 | 0.043 | 25% | 92% | 8% | 208 |
| temp_1.0 | 40 | 0.334 ± 0.018 | 0.777 ± 0.026 | 0.013 | 2% | 100% | 0% | 213 |
| top_k_20 | 40 | 0.298 ± 0.007 | 0.733 ± 0.014 | 0.027 | 8% | 95% | 5% | 238 |
| top_k_50 | 40 | 0.298 ± 0.010 | 0.726 ± 0.014 | 0.024 | 10% | 92% | 8% | 237 |
| top_p_0.90 | 40 | 0.303 ± 0.012 | 0.738 ± 0.017 | 0.029 | 10% | 92% | 8% | 222 |
| top_p_0.95 | 40 | 0.312 ± 0.005 | 0.742 ± 0.021 | 0.026 | 2% | 92% | 8% | 230 |

A story is flagged repetitive when its Rep-4 rate is above 0.057, the 95th percentile of real validation stories.

## Generation speed

Fixed length: 200 tokens, 3 runs, 24 threads.

| Setting | Tokens/s | ms per token | Seconds for a 200-token story |
|---|---|---|---|
| greedy | 160.8 | 6.22 | 1.24 |
| temp_0.7 | 160.0 | 6.25 | 1.25 |
| temp_1.0 | 153.8 | 6.50 | 1.30 |
| top_k_20 | 155.4 | 6.44 | 1.29 |
| top_k_50 | 161.8 | 6.18 | 1.24 |
| top_p_0.90 | 144.8 | 6.91 | 1.38 |
| top_p_0.95 | 140.8 | 7.10 | 1.42 |

## Training

- Hardware: Intel64 Family 6 Model 198 Stepping 2, GenuineIntel, 24 logical CPUs, 63.7 GB RAM, 24 threads, CPU only
- Precision fp32 | batch 8 sequences × 8 accumulation steps = effective batch 64 sequences (16,384 tokens)
- 10,000 steps, 164M tokens seen, data `data/full`
- Training time **9.9 h**, average 4,622 tokens/s, peak memory 1,438 MB (highest value in `log.csv`)

## Error analysis (60 stories, top_p_0.90)

- repetition: 13/60 (22%, auto-flagged)
- incomplete: 4/60 (7%, auto-flagged)
- hit_token_limit: 4/60 (7%, auto-flagged)
- Label contradiction / grammar / entity consistency by hand in `results\generations\small\error_analysis.csv`.

## Example generations

**greedy**

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was barking and running around. Lily was scared and didn't know what to do. She ran inside to tell her mom. Her mom said, "Don't worry, Lily. The dog is just a friendly dog. He just wants to play." Lily felt better and went back outside to play. She saw the dog again and tried to pet it. But the dog was too fast and ran away. Lily was sad and didn't know what to do. She went back inside and told her mom what happened. Her mom said, "Don't worry, Lily. We can get you a new one." Lily was happy again and they went outside to play.

**temp_0.7**

> Once upon a time there was a girl called Lisa. Lisa had a very special toy. It was a shiny car. She loved it! One day, Lisa's mom asked her to help with the laundry. Lisa was so excited to help. She quickly grabbed the car and started to put it in the laundry. When Lisa was done putting the clothes in the washing machine, she was so happy. She said to her mom, "Thank you!" Then she started to clean the laundry. After she was done, Lisa's mom said to her, "See, Lisa? Pasting is the best way to help the laundry!" Lisa smiled and said, "I'm glad I could help!"

**temp_1.0**

> Once upon a time, there was a little girl named Lily. She loved to help her mommy in the kitchen. One day, her mommy asked her to prepare something with some soap. Lily was very excited to help and she went to the kitchen. She opened the headed to the kitchen and started looking for the soap. She saw a lot of soap in the cupboard. She thought, "I will put this in the soap so it fits together." Lily poured the soap into her hands and it matched. She was so proud of her work. "Wow!" she exclaimed. As she stirred the soap, her mommy came in and said, "What are you trying to do, Lily?" Lily told her, "I want to be a fireman!" Her mommy smiled and said, "Well, we can cook something else if we need to." They cooked the soap together and when they were done, they had spaghetti proudly. Lily was amazed by how strong they were. She was proud that she could make a snowman and used all the soap he needed to hairdresser.

**top_k_20**

> Once upon a time there was a very special girl named Emily. She was three years old and loved to play in her garden. One day, Emily wanted to make some soup for dinner. She went to the kitchen and saw that the bowl of soup had turned on a stove. She grabbed a pot and put the soup inside. She stirred it all up, so it smelled very good. Suddenly, her mom stepped out and said, "Anna, the soup is ready. It looks delicious, isn't it? I'm going to try it!" Emily was excited and said, â€œYes! I can taste it!â€ Mommy smiled and said, â€œLetâ€™s go get the soup and then we can make it yummy.â€ So Emily and her mom went to the kitchen together. When they got to the kitchen, Emily was amazed. There were delicious vegetables, vegetables, and salads. They started mixing them together so they could smell. Emily asked, â€œCan I try these, please?â€ Mommy smiled and said, â€œOf course, you can try.â€ They mixed the vegetables and it tasted very delicious. After they were done mixing, Emily took another deep breath and started to try some of the salad. She smiled and said, â€œSee how yummy it is!â€ And from then on, Emily was never feeling too hungry.

**top_k_50**

> Once upon a time, there was a little boy named Timmy. Timmy loved to explore new places. One day, Timmy found a big, shiny rock on the ground. He wondered what it was. Timmy started to search for it, but he couldn't find it. He was very sad. Just then, Timmy's mom came in and said, "Timmy, let's sort the rocks together." She opened his tools and found all the same rocks. They were all different colors and sizes, but they were also part of the shape to other shape. Timmy was so happy that he got to spend his time looking for the rock. From that day on, Timmy always looked for things to sort his own shapes.

**top_p_0.90**

> Once upon a time there was a lazy bear named Lola. She was just walking around the forest, not wanting to help her friends. One day, Lola was walking through the forest when she heard a loud noise. She saw a rabbit near a tree and she knew she had to be brave. She decided to introduce herself to the rabbit. She approached it and said, "Hello! I'm almost so lazy." The rabbit was surprised by Lola's guidance and explained that she was just chasing the rabbit. Lola was amazed and felt a little scared. She asked the rabbit why she was so lazy, but the rabbit replied, "You need to be brave and be brave." Lola was sad, but she knew that the rabbit wanted to be her friend. She said, "Let's be friends, I'll be your friend and be your friend." The rabbit and Lola became friends and ran around the forest together. They ran and played tag, and they were both very happy. From that day on, Lola was always curious and kind, and her friend was so kind and friendly.

**top_p_0.95**

> Once upon a time, there was a soft little bunny named Benny. Benny loved to hop around in the forest with his friends. One day, Benny saw a big, tough carrot. He wanted to eat it, but his friends warned him that it was not safe to eat it. Benny didn't listen to his friends and decided to hop over to get the carrot. When he got there, he saw his friends fighting. Benny told them not to fight because it was hard work and would cause such mistakes. His friends thought about it and decided to help. They all worked together to move the carrot from to front of the carrot. After a few minutes, a strong gust of wind blew and the carrot was able to hold onto the carrot. Benny was happy and grateful to his friends for helping him. From that day on, Benny made sure to always be careful with things he doesn't want to keep.
