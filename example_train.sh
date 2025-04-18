python -m agentarena.training.train \
  --episodes 50000 \
  --model-name mb_basic \
  --save-freq 300 \
  --reward-type basic \
  --learning-rate 0.01 \
  --gamma 0.99 \
  --epsilon 0.9 \
  --epsilon-decay 0.9995 \
  --epsilon-min 0.25
