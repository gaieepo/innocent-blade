import numpy as np
import pygame

from game import Footman, Monk, Rifleman
from utils import FPS, HEIGHT, LANE_LENGTH, VIZ, WIDTH


class Viewer:
    WHITE = (255, 255, 255)
    LIGHT_GRAY = (10, 10, 10)

    def __init__(self, caption):
        pygame.init()
        pygame.display.set_caption(caption)
        self.fps = FPS
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.surface = pygame.Surface(self.screen.get_size())

    def render(self, game, mode='human', close=False, left_action=None, right_action=None):
        self.surface.fill(self.WHITE)

        font = pygame.font.Font(None, 24)
        left_text = font.render(game.left.tag, 1, self.LIGHT_GRAY)
        self.surface.blit(left_text, left_text.get_rect())

        right_text = font.render(game.right.tag, 1, self.LIGHT_GRAY)
        right_textpos = right_text.get_rect()
        right_textpos.right = WIDTH
        self.surface.blit(right_text, right_textpos)

        fps_text = font.render('%d | %d' % (int(self.clock.get_fps()), game.timer), 1, self.LIGHT_GRAY)
        fps_textpos = fps_text.get_rect()
        fps_textpos.midtop = (WIDTH / 2, 0)
        self.surface.blit(fps_text, fps_textpos)

        # render state text
        # (optional) render action
        if left_action is not None and right_action is not None:
            game.left.render_state['action'] = left_action
            game.right.render_state['action'] = right_action

        left_offset = 50

        for k, v in game.left.render_state.items():
            if k != 'army':
                if v in VIZ:
                    k_text = font.render(f'{k}', 1, VIZ[v])
                elif k == 'base':
                    left_base_health_ratio = np.clip(game.left.base.health / game.left.base.max_health, 0, 1)
                    k_text = font.render(
                        f'{k}: {v}', 1, (255 * (1 - left_base_health_ratio), 255 * left_base_health_ratio, 0)
                    )
                else:
                    k_text = font.render(f'{k}: {v}', 1, self.LIGHT_GRAY)
                k_textpos = k_text.get_rect()
                k_textpos.top = left_offset
                self.surface.blit(k_text, k_textpos)
                left_offset += 20

        right_offset = 50

        for k, v in game.right.render_state.items():
            if k != 'army':
                if v in VIZ:
                    k_text = font.render(f'{k}', 1, VIZ[v])
                elif k == 'base':
                    right_base_health_ratio = np.clip(game.right.base.health / game.right.base.max_health, 0, 1)
                    k_text = font.render(
                        f'{k}: {v}', 1, (255 * (1 - right_base_health_ratio), 255 * right_base_health_ratio, 0)
                    )
                else:
                    k_text = font.render(f'{k}: {v}', 1, self.LIGHT_GRAY)
                k_textpos = k_text.get_rect()
                k_textpos.right = WIDTH
                k_textpos.top = right_offset
                self.surface.blit(k_text, k_textpos)
                right_offset += 20

        # render comic army
        for unit in game.left.army:
            if isinstance(unit, Footman):
                pygame.draw.rect(
                    self.surface,
                    (0, 255, 0),
                    pygame.Rect(
                        (
                            50.0 + (WIDTH - 100.0) * (unit.distance / LANE_LENGTH),
                            HEIGHT - 30 * (unit.health / unit.max_health),
                        ),
                        (20, 30 * (unit.health / unit.max_health)),
                    ),
                )
            elif isinstance(unit, Rifleman):
                pygame.draw.rect(
                    self.surface,
                    (0, 255, 255),
                    pygame.Rect(
                        (
                            50.0 + (WIDTH - 100.0) * (unit.distance / LANE_LENGTH),
                            HEIGHT - 50 * (unit.health / unit.max_health),
                        ),
                        (25, 50 * (unit.health / unit.max_health)),
                    ),
                )
            elif isinstance(unit, Monk):
                pygame.draw.rect(
                    self.surface,
                    (0, 120, 120),
                    pygame.Rect(
                        (
                            50.0 + (WIDTH - 100.0) * (unit.distance / LANE_LENGTH),
                            HEIGHT - 50 * (unit.health / unit.max_health),
                        ),
                        (25, 50 * (unit.health / unit.max_health)),
                    ),
                )

        for unit in game.right.army:
            if isinstance(unit, Footman):
                pygame.draw.rect(
                    self.surface,
                    (255, 0, 0),
                    pygame.Rect(
                        (
                            WIDTH - 50.0 - (WIDTH - 100.0) * (unit.distance / LANE_LENGTH),
                            HEIGHT - 30 * (unit.health / unit.max_health),
                        ),
                        (20, 30 * (unit.health / unit.max_health)),
                    ),
                )
            elif isinstance(unit, Rifleman):
                pygame.draw.rect(
                    self.surface,
                    (255, 255, 0),
                    pygame.Rect(
                        (
                            WIDTH - 50.0 - (WIDTH - 100.0) * (unit.distance / LANE_LENGTH),
                            HEIGHT - 50 * (unit.health / unit.max_health),
                        ),
                        (25, 50 * (unit.health / unit.max_health)),
                    ),
                )
            elif isinstance(unit, Monk):
                pygame.draw.rect(
                    self.surface,
                    (0, 120, 120),
                    pygame.Rect(
                        (
                            WIDTH - 50.0 - (WIDTH - 100.0) * (unit.distance / LANE_LENGTH),
                            HEIGHT - 50 * (unit.health / unit.max_health),
                        ),
                        (25, 50 * (unit.health / unit.max_health)),
                    ),
                )

        self.screen.blit(self.surface, (0, 0))

        if mode == 'human':
            pygame.display.flip()

        self.clock.tick(self.fps)

    def set_fps(self, step=None):
        if step is None:
            # reset
            self.fps = FPS
        elif isinstance(step, (int, np.integer)) and step < 0:
            self.fps = 5
        else:
            self.fps = np.clip(self.fps + step, 5, 120)

    def close(self):
        pygame.quit()
