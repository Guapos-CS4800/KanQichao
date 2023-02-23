import deeplake
ds = deeplake.load("hub://activeloop/kuzushiji-kanji")

dataloader = ds.tensorflow()