from agent.agent import Agent


class DefaultAgent(Agent):
    def get_action(self, observation):
        # Beispielstrategie: auf Gegner zulaufen, wenn möglich schießen
        player_pos = observation["player"]
        enemies = observation["enemies"]

        if not enemies:
            return "WAIT"

        target = enemies[0]
        px, py = player_pos
        tx, ty = target

        if abs(px - tx) + abs(py - ty) == 1:
            return "SHOOT"
        elif px < tx:
            return "RIGHT"
        elif px > tx:
            return "LEFT"
        elif py < ty:
            return "DOWN"
        elif py > ty:
            return "UP"
        return "WAIT"
