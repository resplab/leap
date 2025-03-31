1. Run the Simulation
========================

The default simulation configuration options are found in the
`leap/processed_data/config.json <https://resplab.github.io/leap/cli/config.html>`_
file. You can modify these options by creating your own ``config.json`` file and passing it to the
``LEAP`` model.

To run the ``LEAP`` model from the command line, using the default settings:

.. code:: bash

  leap --run-simulation


Simulation Arguments
*********************

.. raw:: html

  <table class="table">
    <thead>
      <tr>
          <th>Flag</th>
          <th>Default</th>
          <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code class="notranslate">--config</code></td>
        <td>
          <a href="https://resplab.github.io/leap/cli/config.html">config.json</a>
        </td>
        <td>
          Path to the <code>config.json</code> file. If none is provided, the default
          <code>config.json</code> will be used.
        </td>
      </tr>

      <tr>
        <td><code class="notranslate">--province</code></td>
        <td><code class="notranslate">"BC"</code></td>
        <td>The province in which to run the simulation. Must be the 2-letter abbreviation for
          the province. One of:
          <ul>
            <li><code class="notranslate">CA</code>: All of Canada</li>
            <li><code class="notranslate">AB</code>: Alberta</li>
            <li><code class="notranslate">BC</code>: British Columbia</li>
            <li><code class="notranslate">MB</code>: Manitoba</li>
            <li><code class="notranslate">NB</code>: New Brunswick</li>
            <li><code class="notranslate">NL</code>: Newfoundland and Labrador</li>
            <li><code class="notranslate">NS</code>: Nova Scotia</li>
            <li><code class="notranslate">NT</code>: Northwest Territories</li>
            <li><code class="notranslate">NU</code>: Nunavut</li>
            <li><code class="notranslate">ON</code>: Ontario</li>
            <li><code class="notranslate">PE</code>: Prince Edward Island</li>
            <li><code class="notranslate">QC</code>: Quebec</li>
            <li><code class="notranslate">SK</code>: Saskatchewan</li>
            <li><code class="notranslate">YT</code>: Yukon</li>
          </ul>
        </td>
      </tr>

      <tr>
        <td><code class="notranslate">--max-age</code></td>
        <td><code class="notranslate">111</code></td>
        <td>The maximum age of a person in the model. For example:
        <code>--max-age 100</code>
        </td>
      </tr>

      <tr>
        <td><code class="notranslate">--min-year</code></td>
        <td><code class="notranslate">2024</code></td>
        <td>The year the simulation starts. For example:
        <code>--min-year 2001</code>
        </td>
      </tr>

      <tr>
        <td><code class="notranslate">--time-horizon</code></td>
        <td><code class="notranslate">13</code></td>
        <td>The number of years to run the simulation for. For example:
        <code>--time-horizon 2</code>
        </td>
      </tr>

      <tr>
        <td><code class="notranslate">--population-growth-type</code></td>
        <td><code class="notranslate">"LG"</code></td>
        <td>The population growth scenario to use. One of:
          <ul>
            <li><code class="notranslate">past</code>: historical data</li>
            <li><code class="notranslate">LG</code>: low-growth projection</li>
            <li><code class="notranslate">HG</code>: high-growth projection</li>
            <li><code class="notranslate">M1</code>: medium-growth 1 projection</li>
            <li><code class="notranslate">M2</code>: medium-growth 2 projection</li>
            <li><code class="notranslate">M3</code>: medium-growth 3 projection</li>
            <li><code class="notranslate">M4</code>: medium-growth 4 projection</li>
            <li><code class="notranslate">M5</code>: medium-growth 5 projection</li>
            <li><code class="notranslate">M6</code>: medium-growth 6 projection</li>
            <li><code class="notranslate">FA</code>: fast-aging projection</li>
            <li><code class="notranslate">SA</code>: slow-aging projection</li>
          </ul>

        See
        <a href="https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm">StatCan</a>.         
        For example:
        <code>--population-growth-type "M3"</code>
        </td>
      </tr>

      <tr>
        <td><code class="notranslate">--num-births-initial</code></td>
        <td><code class="notranslate">100</code></td>
        <td>The number of new babies born in the first year of the simulation. For example:
        <code>--num-births-initial 10</code>
        </td>
      </tr>

      <tr>
        <td><code class="notranslate">--ignore-pollution</code></td>
        <td><code class="notranslate">False</code></td>
        <td>If this flag is used, the simulation will not include air pollution as a factor affecting the model.
        </td>
      </tr>

      <tr>
        <td><code class="notranslate">--path-output</code></td>
        <td>
          <code class="notranslate">PROVINCE-</code><br>
          <code class="notranslate">MAX_AGE-</code><br>
          <code class="notranslate">MIN_YEAR-</code><br>
          <code class="notranslate">TIME_HORIZON-</code><br>
          <code class="notranslate">POPULATION_GROWTH_TYPE-</code><br>
          <code class="notranslate">NUM_BIRTHS_INITIAL</code><br>
        </td>
        <td>The name of the output directory where the results will be saved. For example: <code>--path-output simulation1</code> will save the outputs to <code>LEAP/output/simulation1</code>.
        </td>
        </td>
      </tr>

      <tr>
        <td><code class="notranslate">--force</code></td>
        <td><code class="notranslate">False</code></td>
        <td>If this flag is used, then <code>PATH_OUTPUT</code> will be used as the destination folder without prompting for confirmation, overwriting any existing data located there.
        </td>
      </tr>

      <tr>
        <td><code class="notranslate">--verbose</code></td>
        <td><code class="notranslate">False</code></td>
        <td>If this flag is used, the simulation will print out more information about the
          simulation as it runs. This is useful for debugging purposes.
        </td>
      </tr>
    </tbody>
  </table>


Examples
********

To run the simulation for 1 year, starting in ``2024``, with the maximum age of ``4``,
and ``10`` new borns in the first year:

.. code-block:: bash

  leap --run-simulation --time-horizon 1 --num-births-initial 10 --max-age 4 --min-year 2024 --path-output PATH/TO/OUTPUT


To specify the province and population growth scenario:

.. code-block:: bash

  leap --run-simulation --time-horizon 1 --num-births-initial 10 --max-age 4 --province "CA" --min-year 2024 --population-growth-type "M3" --path-output PATH/TO/OUTPUT


If you would like to use your own ``config.json`` file instead of the default one:

.. code-block:: bash

  leap --run-simulation --config PATH/TO/YOUR/CONFIG.json

