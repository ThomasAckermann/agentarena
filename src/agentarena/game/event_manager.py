"""
Event management system for AgentArena.
"""

from agentarena.models.events import (
    BulletFiredEvent,
    EnemyHitEvent,
    EntityDestroyedEvent,
    PlayerHitEvent,
)


class EventManager:
    def __init__(self) -> None:
        pass

    def create_bullet_fired_event(self, events, game_time, owner_id, direction, position) -> None:
        events.append(
            BulletFiredEvent(
                timestamp=game_time,
                owner_id=owner_id,
                direction=direction,
                position=position,
            ),
        )

    def create_enemy_hit_event(self, events, game_time, enemy_id, x, y) -> None:
        events.append(
            EnemyHitEvent(
                timestamp=game_time,
                enemy_id=enemy_id,
                damage=1,
                position=(x, y),
            ),
        )

    def create_player_hit_event(
        self,
        events,
        game_time,
        bullet_owner,
        x,
        y,
    ) -> None:
        events.append(
            PlayerHitEvent(
                timestamp=game_time,
                damage=1,
                bullet_owner=bullet_owner,
                position=(x, y),
            ),
        )

    def create_entity_destroyed_event(
        self,
        events,
        game_time,
        entity_id,
        entity_type,
        x,
        y,
    ) -> None:
        events.append(
            EntityDestroyedEvent(
                timestamp=game_time,
                entity_id=entity_id,
                entity_type=entity_type,
                position=(
                    x if x is not None else 0,
                    y if y is not None else 0,
                ),
            ),
        )

    def create_player_destroyed_event(self, events, game_time, player) -> None:
        events.append(
            EntityDestroyedEvent(
                timestamp=game_time,
                entity_id="player",
                entity_type="player",
                position=(
                    player.x if player.x is not None else 0,
                    player.y if player.y is not None else 0,
                ),
            ),
        )
