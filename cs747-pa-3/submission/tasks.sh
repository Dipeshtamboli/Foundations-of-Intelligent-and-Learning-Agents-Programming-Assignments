#!bin/bash


# For required tasks
python task_runner.py --num_actions 4 --steps 10000 --alpha 0.6 --epsilon 0.05 --algorithm "sarsa" --plot_save_name "task2_sarsa(0).jpg"
python task_runner.py --num_actions 8 --steps 10000 --alpha 0.6 --epsilon 0.05 --algorithm "sarsa" --plot_save_name "task3_sarsa(0)_king_moves.jpg"
python task_runner.py --num_actions 4 --steps 10000 --alpha 0.6 --epsilon 0.05 --algorithm "sarsa" --stochastic True --plot_save_name "task4_sarsa(0)_four_moves_stoch.jpg"
python task_runner.py --num_actions 8 --steps 10000 --alpha 0.6 --epsilon 0.05 --algorithm "sarsa" --stochastic True --plot_save_name "task4_sarsa(0)_king_moves_stoch.jpg"
python task_runner.py --num_actions 4 --steps 10000 --alpha 0.6 --epsilon 0.05 --algorithm "all" --plot_save_name "task5_all_algos.jpg"

# For generating additional graphs
python additional_plots.py --what_to_plot "sarsa_task_2-3-4_comparison" --plot_save_name "sarsa_task_2-3-4_comparison.jpg"
python additional_plots.py --what_to_plot "episodes_vs_al_epsi" --step_size 0.25 --plot_save_name "Episodes vs al_epsi step-0.25.jpg"