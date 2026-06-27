def reduce_sizes():
    with open('report/final_report.tex', 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply size reductions
    content = content.replace('width=0.6\\textwidth]{report/images/silba-1k.png}', 'width=0.42\\textwidth]{report/images/silba-1k.png}')
    
    content = content.replace('begin{subfigure}[b]{0.48\\textwidth}\n        \\centering\n        \\includegraphics[width=\\textwidth]{report/images/suax-vrx-setup-annotated.png}', 'begin{subfigure}[b]{0.42\\textwidth}\n        \\centering\n        \\includegraphics[width=\\textwidth]{report/images/suax-vrx-setup-annotated.png}')
    
    content = content.replace('begin{subfigure}[b]{0.48\\textwidth}\n        \\centering\n        \\includegraphics[width=\\textwidth]{report/images/small-boat.png}', 'begin{subfigure}[b]{0.42\\textwidth}\n        \\centering\n        \\includegraphics[width=\\textwidth]{report/images/small-boat.png}')

    content = content.replace('height=7cm, width=\\textwidth, keepaspectratio]{report/images/LOFAR_Joint_Signal.png}', 'height=4.8cm, width=\\textwidth, keepaspectratio]{report/images/LOFAR_Joint_Signal.png}')
    content = content.replace('height=7cm, width=\\textwidth, keepaspectratio]{report/images/lofar_annotated.png}', 'height=4.8cm, width=\\textwidth, keepaspectratio]{report/images/lofar_annotated.png}')

    content = content.replace('width=0.7\\textwidth]{report/images/reward_flowchart.png}', 'width=0.5\\textwidth]{report/images/reward_flowchart.png}')
    content = content.replace('width=0.6\\textwidth]{report/images/high_level_architecture.png}', 'width=0.45\\textwidth]{report/images/high_level_architecture.png}')
    
    content = content.replace('width=0.7\\textwidth]{report/images/rl_comparison_croatia_2407_1.png}', 'width=0.55\\textwidth]{report/images/rl_comparison_croatia_2407_1.png}')
    
    content = content.replace('width=0.7\\textwidth]{report/images/croatia_2407_1_double_q_learning_timeline.png}', 'width=0.55\\textwidth]{report/images/croatia_2407_1_double_q_learning_timeline.png}')
    content = content.replace('width=0.7\\textwidth]{report/images/NMF_Components_Joint_Signal.png}', 'width=0.55\\textwidth]{report/images/NMF_Components_Joint_Signal.png}')

    # Also reduce subfigure sizes of convergence plots
    # From:
    #     \begin{subfigure}[b]{0.48\textwidth}
    #         \centering
    #         \includegraphics[width=\textwidth]{report/images/convergence_combined_500.png}
    # To:
    #     \begin{subfigure}[b]{0.42\textwidth}
    content = content.replace('begin{subfigure}[b]{0.48\\textwidth}\n        \\centering\n        \\includegraphics[width=\\textwidth]{report/images/convergence_combined_500.png}', 'begin{subfigure}[b]{0.42\\textwidth}\n        \\centering\n        \\includegraphics[width=\\textwidth]{report/images/convergence_combined_500.png}')
    content = content.replace('begin{subfigure}[b]{0.48\\textwidth}\n        \\centering\n        \\includegraphics[width=\\textwidth]{report/images/convergence_individual_500.png}', 'begin{subfigure}[b]{0.42\\textwidth}\n        \\centering\n        \\includegraphics[width=\\textwidth]{report/images/convergence_individual_500.png}')

    with open('report/final_report.tex', 'w', encoding='utf-8') as f:
        f.write(content)

    print("LaTeX image sizes reduced successfully!")

if __name__ == "__main__":
    reduce_sizes()
