python -m agentarena.training.train \
  --episodes 50000 \
  --model-name mb_basic \
  --save-freq 300 \
  --reward-type advanced \
  --learning-rate 0.001 \
  --gamma 0.99 \
  --epsilon 0.3 \
  --epsilon-decay 0.995 \
  --epsilon-min 0.10
