python -m agentarena.training.train \
  --episodes 5000 \
  --model-name m3pro_basic \
  --save-freq 200 \
  --reward-type basic \
  --learning-rate 0.005 \
  --gamma 0.97 \
  --epsilon 0.9 \
  --epsilon-decay 0.9993 \
  --epsilon-min 0.15
