import numpy as np
import pygame

from utils import FPS, HEIGHT, LANE_LENGTH, VIZ, WIDTH, Footman, Monk, Rifleman


class Viewer:
    def __init__(self, caption):
        pygame.init()
        pygame.display.set_caption(caption)
        self.fps = FPS
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.surface = pygame.Surface(self.screen.get_size())

    def render(
        self,
        game,
        mode='human',
        close=False,
        white_action=None,
        black_action=None,
    ):
        self.surface.fill((255, 255, 255))

        font = pygame.font.Font(None, 24)
        white_text = font.render('White', 1, (10, 10, 10))
        self.surface.blit(white_text, white_text.get_rect())

        black_text = font.render('Black', 1, (10, 10, 10))
        black_textpos = black_text.get_rect()
        black_textpos.right = WIDTH
        self.surface.blit(black_text, black_textpos)

        fps_text = font.render(
            '%d | %d' % (int(self.clock.get_fps()), game.timer),
            1,
            (10, 10, 10),
        )
        fps_textpos = fps_text.get_rect()
        fps_textpos.midtop = (WIDTH / 2, 0)
        self.surface.blit(fps_text, fps_textpos)

        # render state text
        # (optional) render action
        if white_action is not None and black_action is not None:
            game.white.render_state['action'] = white_action
            game.black.render_state['action'] = black_action

        white_offset = 50

        for k, v in game.white.render_state.items():
            if k != 'army':
                if v in VIZ:
                    k_text = font.render(f'{k}', 1, VIZ[v])
                elif k == 'base':
                    white_base_health_ratio = np.clip(
                        game.white.base.health / game.white.base.max_health,
                        0,
                        1,
                    )
                    k_text = font.render(
                        f'{k}: {v}',
                        1,
                        (
                            255 * (1 - white_base_health_ratio),
                            255 * white_base_health_ratio,
                            0,
                        ),
                    )
                else:
                    k_text = font.render(f'{k}: {v}', 1, (10, 10, 10))
                k_textpos = k_text.get_rect()
                k_textpos.top = white_offset
                self.surface.blit(k_text, k_textpos)
                white_offset += 20

        black_offset = 50

        for k, v in game.black.render_state.items():
            if k != 'army':
                if v in VIZ:
                    k_text = font.render(f'{k}', 1, VIZ[v])
                elif k == 'base':
                    black_base_health_ratio = np.clip(
                        game.black.base.health / game.black.base.max_health,
                        0,
                        1,
                    )
                    k_text = font.render(
                        f'{k}: {v}',
                        1,
                        (
                            255 * (1 - black_base_health_ratio),
                            255 * black_base_health_ratio,
                            0,
                        ),
                    )
                else:
                    k_text = font.render(f'{k}: {v}', 1, (10, 10, 10))
                k_textpos = k_text.get_rect()
                k_textpos.right = WIDTH
                k_textpos.top = black_offset
                self.surface.blit(k_text, k_textpos)
                black_offset += 20

        # render comic army
        for unit in game.white.army:
            if isinstance(unit, Footman):
                pygame.draw.rect(
                    self.surface,
                    (0, 255, 0),
                    pygame.Rect(
                        (
                            50.0
                            + (WIDTH - 100.0) * (unit.distance / LANE_LENGTH),
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
                            50.0
                            + (WIDTH - 100.0) * (unit.distance / LANE_LENGTH),
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
                            50.0
                            + (WIDTH - 100.0) * (unit.distance / LANE_LENGTH),
                            HEIGHT - 50 * (unit.health / unit.max_health),
                        ),
                        (25, 50 * (unit.health / unit.max_health)),
                    ),
                )

        for unit in game.black.army:
            if isinstance(unit, Footman):
                pygame.draw.rect(
                    self.surface,
                    (255, 0, 0),
                    pygame.Rect(
                        (
                            WIDTH
                            - 50.0
                            - (WIDTH - 100.0) * (unit.distance / LANE_LENGTH),
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
                            WIDTH
                            - 50.0
                            - (WIDTH - 100.0) * (unit.distance / LANE_LENGTH),
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
                            WIDTH
                            - 50.0
                            - (WIDTH - 100.0) * (unit.distance / LANE_LENGTH),
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
