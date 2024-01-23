# 对于syncnet
nohup python  train_syncnet_sam.py > syncnet.log 2>&1 &
nohup python  hq_wav2lip_sam_train.py > wav2lip_resume.log 2>&1 &