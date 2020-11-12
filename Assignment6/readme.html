<!DOCTYPE html>
<html>

<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HMM</title>
  <link rel="stylesheet" href="https://stackedit.io/style.css" />
</head>

<body class="stackedit">
  <div class="stackedit__html"><h1 id="hidden-markov-models-50-points">Hidden Markov Models (50 points)</h1>

<h2 id="Instructions">General instructions</a></h2><ul>
<li>In this task you will implement various inference algorithms for <strong>HMM</strong> and also apply them to sentence tagging. We provide the bootstrap code and you are expected to complete the functions.</li>
<li>Do not import libraries other than those already imported in the original code.</li>
<li>Please follow the type annotations. You have to make the function’s return values match the required type.</li>
<li>Only modifications in files {<code>hmm.py</code>, <code>tagger.py</code>} in the "work" directory will be accepted and graded. All other modifications will be ignored. You can work directly on Vocareum, or download all files from "work", code in your own workspace, and then upload the changes (recommended). </li>
<li>Click the Submit button when you are ready to submit your code for auto-grading. Your final grade is determined by your <strong>last</strong> submission. </li>
</ul>    
  


<h2 id="implementation-30-points">Q1 Implement the inference algorithms</h2>
In <code>hmm.py</code>, you will find a class called HMM whose attributes specify the model parameters of a Hidden Markov Model (including its initial state probability, transition probability, and emission probability).
You need to implement the following six functions
<ul>
<li><code>forward</code>: compute the forward messages</li>
<li><code>backward</code>: compute the backward messages</li>
<li><code>sequence_prob</code>: compute the probability of observing a particular sequence</li>
<li><code>posterior_prob</code>: compute the probability of the state at a particular time step given the observation sequence </li>
<li><code>likelihood_prob</code>: compute the probability of state transition at a particular time step given the observation sequence</li>
<li><code>viterbi</code>: compute the most likely hidden state path using the Viterbi algorithm.</li>
</ul>
We have discussed how to compute all these via dynamic programming in the lecture.
Here, the only thing you need to pay extra attention to is that the indexing system is slightly different between the python code and the formulas we discussed (the former starts from 0 and the latter starts from 1).
Read the comments in the code carefully to get a better sense of this discrepancy.



<h2 id="application-to-speech-tagging--20-points">Q2 Application to speech tagging </h2>
<p>Part-of-Speech (POS) is a category of words (or, more generally, of lexical items) which have similar grammatical properties. (Example: noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection, and sometimes numeral, article, or determiner.)
Part-of-Speech Tagging (POST) is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech, based on both its definition and its context.</p>
<p>Here you will use HMM to perform POST, where the tags are states and the words are observations. 
We collect our dataset and tags with the Dataset class. Dataset class includes tags, train_data and test_data. Both datasets include a list of sentences, and each sentence is an object of the Line class. 
You only need to implement the <code>model_training</code> function and the <code>speech_tagging</code> function in <code>tagger.py</code>.

<ul>
<li><code>model_training</code>: in this function, you need to build an instance of the HMM class by setting its five parameters <code>(pi, A, B, obs_dict, state_dict)</code>. 
The way you estimate the parameter <code>pi, A, B</code> is simply by counting the corresponding frequency from the given training set, as we discussed in the class. Read the comments in the code for more instructions.
</li>
<li><code>speech_tagging</code>: given the HMM built from model_training, now your task is to run the Viterbi algorithm to find the most likely tagging of a given sentence.
One particular detail you need to take care of is when you meet a new word which was unseen in the training dataset.
In this case, you need to update the dictionary <code>obs_dict</code> accordingly, and also expand the emission matrix by assuming that the probability of seeing this new word under any state is 1e-6.
Again, read the comments in the code for more instructions.
</li>
</ul>

<h2 id="grading-guideline">Q3 Testing</h2>
Once you finish these two parts, run <code>hmm_test_script.py</code>.
We will first run all your inference algorithms on a toy HMM model specified in <code>hmm_model.json</code>,
and then also your tagging code on the dataset stored in <code>pos_sentences.txt</code> and <code>pos_tags.txt</code>.
In both cases, the script tells you what your outputs vs. the correct outputs are.

<h2 id="grading-guideline">Grading guideline</h2>
<p>1 Inference algorithms (30 points)</p>
<ol>
<li><code>forward</code> function - 5 = 5x1 points</li>
<li><code>backward</code> function - 5 = 5x1 points</li>
<li><code>sequence_prob</code> function - 2.5 = 5x0.5 points</li>
<li><code>posterior_prob</code> function - 5 = 5x1 points</li>
<li><code>likelihood_prob</code> function - 5 = 5x1 points</li>
<li><code>viterbi</code> function - 7.5 = 5*1.5 points</li>
</ol>
<p>There are 5 sets of grading data used to initialize the HMM class and test your functions. To receive full credits, your output of functions 1-5 should be within an error of 1e-6, and your output of the viterbi function should be identical with ours.</p>

<p>2 Application to Part-of-Speech Tagging (20 points)</p>
<ol>
<li><code>model_training</code> - 10 = 10x(your_correct_pred_cnt/our_correct_pred_cnt)</li>
<li><code>speech_tagging</code> - 10 = 10x(your_correct_pred_cnt/our_correct_pred_cnt)</li>
</ol>
<p>We will use the dataset given to you for grading this part (with a different random seed). We will train your model and our model on same train_data. <code>model_training</code> function and <code>speech_tagging</code> function will be tested separately.</p>
<p>In order to check your model_training function, we will use 50 sentences from <code>train_data</code> to do Part-of-Speech Tagging (your model + our tagging function vs. our model + our tagging function). To receive full credits, your prediction accuracy should be identical or better than ours.</p>
<p>In order to check your <code>speech_tagging</code> function, we will use 50 sentences from <code>test_data</code> to do Part-of-Speech Tagging (your model + your tagging function vs. our model + our tagging function). Again, to receive full credits, your prediction accuracy should be identical or better than ours.</p>

</div>
</body>

</html>
