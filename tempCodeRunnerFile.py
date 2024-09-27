 plt.subplots()
    enlargeX = 1.2
    x_points = np.arange(len(allmaes)) * enlargeX
    for i,posx in enumerate(x_points):
        ax.bar(posx, allmaes[i], alpha=opacity, color = colors[0], yerr=stderror[i], capsize=5)

    plt.xlabel('Participant')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error between 2D and 3D')
    plt.xticks(x_points, np.arange(1, len(allmaes)+1))


    plt.savefig('plots/mae_per_participant.png', dpi=300, bbox_inches='tight')
    plt.savefig('plots/mae_per_participant.pdf', dpi=300, bbox_inches='tight')
    plt.show()