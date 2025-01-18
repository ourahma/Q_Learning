import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

grid_size = 10
env = np.zeros((grid_size, grid_size))  # Grille vide
goal = (9, 9)
actions = ['H', 'B', 'G', 'D']  # Haut, Bas, Gauche, Droite

# Ajout d'obstacles
obstacles = [(0, 5), (0, 6), (3, 3), (3, 4), (6, 5), (6, 6)]
for obs in obstacles:
    env[obs] = 1
    
visited_states=set()



# Définir les récompenses
def get_reward(state,env,strawberries):
    x, y = state
    if env[x, y] == 1:  # Obstacle
        return -10
    elif (x, y) == (9, 9):  # Objectif
        return 100 
    elif env[x, y] == 2:  # Fraise
        env[x, y] = 0  # Consommer la fraise
        return 20
    elif state in visited_states:  # Pénalité pour revisiter
        return -5
    else:
        return -1  # Pénalité par défaut pour éviter les détours inutiles

# Déplacer le robot
def take_action(state, action):
    x, y = state
    
    # Restriction pour la position (0, 0)
    if state == (0, 0):
        if action == 'B':  # Bas
            x += 1
        elif action == 'D':  # Droite
            y += 1
        # Les autres actions ne sont pas permises, rester dans l'état actuel
        return (x, y)
    
    
    if action == 'H' and x > 0:
        x -= 1
    elif action == 'B' and x < grid_size - 1:
        x += 1
    elif action == 'G' and y > 0:
        y -= 1
    elif action == 'D' and y < grid_size - 1:
        y += 1
    return (x, y)


# Q-learning
def q_learning(episodes,alpha,gamma,epsilon,q_table,strawberries=None):

    for episode in range(episodes):
        state = (0, 0)
        visited_states.clear()  # Réinitialiser les états visités par épisode

        while state != (9, 9):
            # Choisir une action
            if np.random.uniform(0, 1) < epsilon:
                action_index = np.random.choice(len(actions))  # Exploration
            else:
                action_index = np.argmax(q_table[state[0], state[1], :])  # Exploitation
            
            action = actions[action_index]
            visited_states.add(state)
            next_state = take_action(state, action)
            reward = get_reward(next_state,env,strawberries)
            
            # Mettre à jour la Q-valeur
            q_table[state[0], state[1], action_index] += alpha * (
                reward + gamma * np.max(q_table[next_state[0], next_state[1], :]) - 
                q_table[state[0], state[1], action_index]
            )
            
            state = next_state
        
        # Réduire epsilon au fil des épisodes pour diminuer l'exploration
        epsilon = max(0.1, epsilon * 0.99)  # Valeur minimale d'exploration
    return q_table



def visualize_path(ax,env, path,title,strawberries=None):
    grid_size = env.shape[0]
    
    # Dessiner les cases
    for x in range(grid_size):
        for y in range(grid_size):
            if env[x, y] == 1:  # Obstacle
                rect = patches.Rectangle((y, grid_size - x - 1), 1, 1, facecolor='black')
            else:  # Case valide
                rect = patches.Rectangle((y, grid_size - x - 1), 1, 1, edgecolor='gray', facecolor='white')
            ax.add_patch(rect)
    
    # Dessiner le chemin
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        # Matplotlib dessine en coordonnées inversées (x devient y et y devient x)
        ax.plot(
            [start[1] + 0.5, end[1] + 0.5], 
            [grid_size - start[0] - 0.5, grid_size - end[0] - 0.5], 
            color='blue', linewidth=2
        )
    if strawberries:
        for strawberry in strawberries:
            x, y = strawberry
            rect = patches.Rectangle((y, grid_size - x - 1), 1, 1, edgecolor='gray', facecolor='#febcd3')
            ax.add_patch(rect)
    
    # Marquer le point de départ et l'objectif
    start = path[0]
    goal = path[-1]
    ax.plot(start[1] + 0.5, grid_size - start[0] - 0.5, 'go', markersize=10, label="Départ")  # Vert
    ax.plot(goal[1] + 0.5, grid_size - goal[0] - 0.5, 'ro', markersize=10, label="Objectif")  # Rouge
    
    # Configurer les axes
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    
    ax.set_yticklabels(range(grid_size - 1, -1, -1))
    
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title(title)


# Fonction pour visualiser les valeurs Q
def visualize_q_table(Q_table):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.set_xticks(np.arange(Q_table.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(Q_table.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    # Dessiner la grille
    ax.imshow(np.zeros(Q_table.shape[:2]), cmap="gray", origin="upper", alpha=0.3)
    
    # Afficher les valeurs Q pour chaque action
    for x in range(Q_table.shape[0]):
        for y in range(Q_table.shape[1]):
            for action, direction in zip(range(4), ["↑", "↓", "←", "→"]):
                dx, dy = [-0.25, 0.25, 0, 0], [0, 0, -0.25, 0.25]  # Position des actions dans chaque case
                ax.text(
                    y + dy[action],
                    x + dx[action],
                    f"{Q_table[x, y, action]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="blue"
                )
    
    ax.set_xticks(range(Q_table.shape[1]))
    ax.set_yticks(range(Q_table.shape[0]))
    ax.set_xticklabels(range(Q_table.shape[1]))
    ax.set_yticklabels(range(Q_table.shape[0]))
    ax.set_title("Table Q (Valeurs pour chaque Action)")
    plt.tight_layout()
    plt.show()


# Fonction pour visualiser les valeurs optimales (max(Q))
def visualize_Heatmap_optimal_path(Q_table):
    # Calculer les valeurs maximales de Q pour chaque état
    optimal_values = np.max(Q_table, axis=2)
    
    # Créer une heatmap pour représenter les valeurs maximales
    plt.figure(figsize=(8, 8))
    plt.imshow(optimal_values, cmap="viridis", origin="upper", interpolation="none")
    
    # Afficher les valeurs dans chaque cellule
    for x in range(optimal_values.shape[0]):
        for y in range(optimal_values.shape[1]):
            plt.text(y, x, f"{optimal_values[x, y]:.1f}", ha="center", va="center", color="white")
    
    plt.title("Chemin Optimal (Heatmap des valeurs maximales Q)")
    plt.colorbar(label="Valeurs maximales Q")
    plt.xticks(range(optimal_values.shape[1]))
    plt.yticks(range(optimal_values.shape[0]))
    plt.grid(color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.show()



## méthode pour trouver le chemin optimal
def find_optimal_path(q_table, source, objective):
    # Initialisation
    current_state = source
    optimal_path = [current_state]

    while current_state != objective:
        # Trouver l'action avec la Q-valeur maximale
        action_index = np.argmax(q_table[current_state[0], current_state[1], :])
        action = actions[action_index]

        # Calculer l'état suivant
        next_state = take_action(current_state, action)

        # Vérifier si l'état suivant est valide (pas un obstacle)
        if env[next_state[0], next_state[1]] == 1:
            print(f"Erreur : Le chemin passe par un obstacle à {next_state} !")
            break

        # Ajouter l'état suivant au chemin
        optimal_path.append(next_state)
        current_state = next_state

    # Ajouter l'objectif au chemin
    if current_state == objective:
        optimal_path.append(objective)

    return optimal_path