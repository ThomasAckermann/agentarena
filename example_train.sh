python -m agentarena.training.train \
  --episodes 5000 \
  --model-name m3pro_enhanced_reward \
  --save-freq 200 \
  --reward-type enhanced \
  --learning-rate 0.001 \
  --gamma 0.60 \
  --epsilon 1 \
  --epsilon-decay 0.9997 \
  --epsilon-min 0.15

