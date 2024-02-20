
<title>MNIST Classification Models</title>
<h1>MNIST Classification Models</h1>
<p>This repository contains PyTorch implementations of multiple models for MNIST digit classification.</p>

<h2>Models Overview:</h2>
<table>
    <tr>
        <th>Network No.</th>
        <th>Num of Epochs</th>
        <th>Test Accuracy</th>
        <th>Training Time (seconds)</th>
        <th>Model Capacity</th>
        <th>Architecture</th>
    </tr>
    <tr>
        <td>1</td>
        <td>20</td>
        <td>97.37%</td>
        <td>5.4</td>
        <td>48859</td>
        <td>FC (ReLU) - FC</td>
    </tr>
    <tr>
        <td>2</td>
        <td>25</td>
        <td>85.27%</td>
        <td>15.5</td>
        <td>49111</td>
        <td>FC (ReLU) - FC</td>
    </tr>
    <tr>
        <td>3</td>
        <td>23</td>
        <td>85.57%</td>
        <td>31.3</td>
        <td>49319</td>
        <td>FC (ReLU) - FC (ReLU) - FC (ReLU) - FC</td>
    </tr>
    <tr>
        <td>4</td>
        <td>40</td>
        <td>89.84%</td>
        <td>352</td>
        <td>45607</td>
        <td>CNN (ReLU) - Dropout - MaxPooling - CNN (ReLU) - Dropout - MaxPooling - CNN (ReLU) - MaxPooling - FC</td>
    </tr>
</table>

<h2>Instructions to Run:</h2>
<ol>
    <li>Clone the repository:</li>
    <pre><code>git clone https://github.com/your-username/your-repository.git</code></pre>
    <li>Navigate to the cloned repository:</li>
    <pre><code>cd your-repository</code></pre>
    <li>Install the required dependencies. Ensure you have PyTorch installed:</li>
    <pre><code>pip install torch torchvision</code></pre>
    <li>Run the desired model script. For example, to run Network 1:</li>
    <pre><code>python network1.py</code></pre>
</ol>

<footer>
    <p>For any issues or inquiries, please contact Faisal Omari (325616894) or Saji Assi (314831207).</p>
</footer>
