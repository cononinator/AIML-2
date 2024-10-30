import numpy as np
import matplotlib.pyplot as plt

def manhattan_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def minkowski_distance(p1, p2, k):
    return (np.sum(np.abs(p1 - p2) ** k)) ** (1/k)

def chebyshev_distance(p1, p2):
    return np.max(np.abs(p1 - p2))

# Points
point_A = np.array([1, 1])
point_B = np.array([2, 0])

# Calculate distances
manhattan_dist = manhattan_distance(point_A, point_B)
euclidean_dist = euclidean_distance(point_A, point_B)
minkowski_k3_dist = minkowski_distance(point_A, point_B, 3)
chebyshev_dist = chebyshev_distance(point_A, point_B)

# Set up plot
plt.figure(figsize=(8, 8))
plt.scatter(*point_A, color="black", label="Point A (1, 1)", zorder=5)
plt.scatter(*point_B, color="purple", label="Point B (2, 1)", zorder=5)

# Create points for drawing the contours
theta = np.linspace(0, 2*np.pi, 1000)
r = np.linspace(0, 2.5, 100)
theta_grid, r_grid = np.meshgrid(theta, r)
x_grid = r_grid * np.cos(theta_grid) + point_A[0]
y_grid = r_grid * np.sin(theta_grid) + point_A[1]

# Calculate distances for each point in the grid
distances = np.zeros_like(x_grid)
for i in range(x_grid.shape[0]):
    for j in range(x_grid.shape[1]):
        point = np.array([x_grid[i,j], y_grid[i,j]])
        distances[i,j] = minkowski_distance(point_A, point, 3)

# Plot the contour for Minkowski distance
plt.contour(x_grid, y_grid, distances, levels=[minkowski_k3_dist],
            colors='green', linestyles='--', label="Minkowski (k=3) Distance")

# Draw Euclidean circle
circle_euclidean = plt.Circle(point_A, euclidean_dist, color='blue',
                            fill=False, linestyle='--', label="Euclidean Distance")
plt.gca().add_patch(circle_euclidean)

# Draw Chebyshev square
chebyshev_rect = plt.Rectangle((point_A[0] - chebyshev_dist, point_A[1] - chebyshev_dist),
                              2 * chebyshev_dist, 2 * chebyshev_dist,
                              color='red', fill=False, linestyle='--',
                              label="Chebyshev Distance", linewidth=2)
plt.gca().add_patch(chebyshev_rect)

# Draw Manhattan path
plt.plot([point_A[0], point_B[0]], [point_A[1], point_A[1]], 'orange',
         linestyle='-', label="Manhattan Path", zorder=3)
plt.plot([point_B[0], point_B[0]], [point_A[1], point_B[1]], 'orange',
         linestyle='-', zorder=3)

# Draw direct line
plt.plot([point_A[0], point_B[0]], [point_A[1], point_B[1]], 'purple',
         linestyle='-', label="Line to Point B")

# Add distance labels
plt.text(1.5, 1.05, f"Manhattan: {manhattan_dist}", color="orange")
plt.text(1.5, 0.95, f"Euclidean: {euclidean_dist:.2f}", color="blue")
plt.text(1.5, 0.85, f"Minkowski (k=3): {minkowski_k3_dist:.2f}", color="green")
plt.text(1.5, 0.75, f"Chebyshev: {chebyshev_dist}", color="red")
plt.text(1.05, 1.05, "A", color="black")
plt.text(2.05, -0.05, "B", color="purple")

# Plot configuration
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Distances from Point A to Point B for Different Metrics")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.xlim(-0.5, 2.5)
plt.ylim(-0.5, 2.5)

plt.show()