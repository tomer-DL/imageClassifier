import Augmentor

p = Augmentor.Pipeline("itay_aug_test")
p.rotate(probability=1.0, max_left_rotation=10, max_right_rotation=10)
p.skew_tilt(probability=1.0, magnitude=0.1)
p.random_distortion(probability=0.8, grid_width=3, grid_height=3, magnitude=1)
p.sample(10)
