should i resize to 66x200? my size is (160 - 70 - 20 by 320) == 70x320. 70 is still taller than 66, so the original strides are not too large.

batch size 128 is arbitrarily chosen. it was known to be appropriate for lenet. TODO calculate num trainable variables in lenet vs nvidia.

added dropout; otherwise didn't change nvidia model.

steering correction = 0.1, then 0.2

checkpoint -> save each epoch

generation was limited to flipping, b/c it was cheap to acquire more training samples.

videos/2017-11-08-recovery0/video.mp4
  left swerve at 01:01 again at 02:31
  right swerve at 01:07 again at 02:37
  above repeated mistake suggests that maybe the model has memorized the route. but this may not be true, b/c i have much more data for the challenging track yet the model hasn't memorized it.
  there does not seem to be consistent bias towards driving on the left side of the road or the right side.

did not use pooling

did not customize adam

perhaps owing to the volume of datasets, loss was quite low from the beginning (?)

model visualization
  requirements are py34 and h5py
  while creating a new conda env, also used latest keras which is 2.0.9
