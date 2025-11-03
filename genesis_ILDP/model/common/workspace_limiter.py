import torch

class WorkspaceLimiter:
    def __init__(self,
                 mode: str = 'circular',
                 radius_inner = 0.15,
                 radius_outer = 0.73,
                 soft_loss_coeff = 1e-1,
                 angle_min = -torch.pi * 2 / 2,
                 angle_max = torch.pi * 1 / 2,
                 x_min = None,
                 x_max = None,
                 y_min = None,
                 y_max = None,
                 ):
        self.mode = mode
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        self.soft_loss_coeff = soft_loss_coeff
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def _loss_circular(self, pred_action: torch.Tensor) -> float | torch.Tensor:
        action_radius = torch.sqrt(pred_action[:, :, 0] ** 2 + pred_action[:, :, 1] ** 2)
        action_angle = torch.atan2(pred_action[:, :, 1], pred_action[:, :, 0])

        action_angle = torch.where(action_angle < 0, action_angle + 2 * torch.pi, action_angle)

        dist_violation_inner = torch.clamp(self.radius_inner - action_radius, min=0)
        dist_violation_outer = torch.clamp(action_radius - self.radius_outer, min=0)

        workspace_size = self.radius_outer - self.radius_inner
        dist_diff_normalized = (dist_violation_inner + dist_violation_outer) / workspace_size

        angle_min_normalized = self.angle_min if self.angle_min >= 0 else self.angle_min + 2 * torch.pi
        angle_max_normalized = self.angle_max if self.angle_max >= 0 else self.angle_max + 2 * torch.pi

        angle_violation_min = torch.clamp(angle_min_normalized - action_angle, min=0)
        angle_violation_max = torch.clamp(action_angle - angle_max_normalized, min=0)

        angle_range = angle_max_normalized - angle_min_normalized
        angle_diff_normalized = (angle_violation_min + angle_violation_max) / angle_range

        combined_violation = dist_diff_normalized + angle_diff_normalized

        soft_loss = self.soft_loss_coeff * combined_violation.mean(dim=-1)

        return soft_loss

    def _loss_rectangular(self, pred_action: torch.Tensor) -> float | torch.Tensor:
        x_violation_min = torch.clamp(self.x_min - pred_action[:, :, 0], min=0)
        x_violation_max = torch.clamp(pred_action[:, :, 0] - self.x_max, min=0)
        y_violation_min = torch.clamp(self.y_min - pred_action[:, :, 1], min=0)
        y_violation_max = torch.clamp(pred_action[:, :, 1] - self.y_max, min=0)

        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min

        x_diff_normalized = (x_violation_min + x_violation_max) / x_range
        y_diff_normalized = (y_violation_min + y_violation_max) / y_range

        combined_violation = x_diff_normalized + y_diff_normalized

        soft_loss = self.soft_loss_coeff * combined_violation.mean(dim=-1)

        return soft_loss

    def _loss(self, pred_action: torch.Tensor) -> float | torch.Tensor:
        if self.mode == 'circular':
            return self._loss_circular(pred_action)
        elif self.mode == 'rectangular':
            return self._loss_rectangular(pred_action)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _limit_circular(self, pred_action):
        action_radius = torch.sqrt(pred_action[:, :, 0] ** 2 + pred_action[:, :, 1] ** 2)
        action_angle = torch.atan2(pred_action[:, :, 1], pred_action[:, :, 0])

        angle_min_normalized = self.angle_min if self.angle_min >= 0 else self.angle_min + 2 * torch.pi
        angle_max_normalized = self.angle_max if self.angle_max >= 0 else self.angle_max + 2 * torch.pi

        clamped_radius = torch.clamp(action_radius, min=self.radius_inner, max=self.radius_outer)
        clamped_angle = torch.clamp(action_angle, min=angle_min_normalized, max=angle_max_normalized)

        return torch.stack([clamped_radius * torch.cos(clamped_angle),
                            clamped_radius * torch.sin(clamped_angle)], dim=-1)

    def _limit_rectangular(self, pred_action):
        clamped_x = torch.clamp(pred_action[:, :, 0], min=self.x_min, max=self.x_max)
        clamped_y = torch.clamp(pred_action[:, :, 1], min=self.y_min, max=self.y_max)
        return torch.stack([clamped_x, clamped_y], dim=-1)

    def _limit(self, pred_action):
        if self.mode == 'circular':
            return self._limit_circular(pred_action)
        elif self.mode == 'rectangular':
            return self._limit_rectangular(pred_action)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def clip(self, pred_action):
        return self._limit(pred_action)

    def limit(self, pred_action):
        return self._loss(pred_action), self._limit(pred_action)
    
    
if __name__ == '__main__':
    pred_action = torch.randn(size=(10, 16, 2)) * 0.1
    Limiter = WorkspaceLimiter()
    print(Limiter.limit(pred_action)[0].shape)