from sample import Sample

sample = Sample(100, [0.44, 0.13, 41], 189)

data = sample.get_sample()
sample.save_sample(data)
