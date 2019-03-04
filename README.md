This is a VideoQA baseline implemented by pytorch.
I just use the most elementary idea to write that code.
But there still exists some places to be improved.

## model architecture
frame_countXframe_feature_vector------->>lstm  get final video cell state    
question_lengthXword_embedding------->>lstm  get final question cell state    
video cell state------->>linear get video fuse part     
question cell state------->>linear get question fuse part  
dot product of above two part--------->>linear------->>as input of decoder(lstm)  