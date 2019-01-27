# Attention-Mechanism
Attention Model for translating dates of various formats to machine readable format.

If the above notebook does not display for some reason, please copy the link of the notebook by right-clicking on it,
go to this free notebook viewing website, https://nbviewer.jupyter.org/ and paste the link over there to open it.

This notebook is based on my project towards earning the Deep Learning Specialization Certificate at Coursera.Org taught by Andrew Ng.


The model used here for translating is the Attention Mechanism. It uses the fact that nearby words have a bigger effect on translation of the word that the farther words.

Future Work:
- Currently I am working on implementing the Actor-Critic Methods with Attention-Mechanism for an effective translation.
- Instead of just using the attention mechanism as the Actor, I would like to use the Critic model based on the BLEU score to manage the variance on translation.
- This will reduce the variance in predictions and overfitting.
- I would like to use the critic model to generate the Q_Next and call the BLEU score as the reward.
- Create a new class like an Environment in the typical Reinforcement learning task, call its reward the BLEU score and next_state as the next word to translate.
-  This way the Actor-Critic Method can be used to represent and train any task.
- To improve the efficiency and reduce over-fitting.
