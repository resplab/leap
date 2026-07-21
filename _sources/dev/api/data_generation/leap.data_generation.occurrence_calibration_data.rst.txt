Occurrence Calibration Data
===============================


To run the data generation for the incidence/prevalence calibration:

.. code-block:: bash

   cd LEAP
   python3 leap/data_generation/occurrence_calibration_data.py


This will update the file 
`leap/processed_data/asthma_occurrence_correction.csv
<https://github.com/resplab/leap/blob/main/leap/processed_data/asthma_occurrence_correction.csv>`_.
This file contains the incidence / prevalence correction parameters and is formatted as follows:

.. raw:: html

  <table class="table">
    <thead>
      <tr>
          <th>Column</th>
          <th>Type</th>
          <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code class="notranslate">year</code></td>
        <td>
          <code class="notranslate">int</code>
        </td>
        <td>
          format <code>XXXX</code>, e.g <code>2000</code>, range <code>[2000, 2026]</code>
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">age</code></td>
        <td>
          <code class="notranslate">int</code>
        </td>
        <td>
          The age in years, a value in <code class="notranslate">[3, 63]</code>
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">sex</code></td>
        <td>
          <code class="notranslate">str</code>
        </td>
        <td>
          <code class="notranslate">"F"</code> = Female,
          <code class="notranslate">"M"</code> = Male
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">correction</code></td>
        <td>
          <code class="notranslate">float</code>
        </td>
        <td>
          The correction term for the asthma incidence / prevalence equation
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">type</code></td>
        <td>
          <code class="notranslate">str</code>
        </td>
        <td>
          One of <code class="notranslate">"incidence"</code>
          <code class="notranslate">"prevalence"</code>
        </td>
      </tr>
    </tbody>
    </table>


If you want to rerun the optimization for the :math:`\beta` parameters, add the flag
``--retrain-beta``:


.. code-block:: bash

   cd LEAP
   python3 leap/data_generation/occurrence_calibration_data.py --retrain-beta



This will update the file `leap/processed_data/asthma_occurrence_correction.csv
<https://github.com/resplab/leap/blob/main/leap/processed_data/asthma_occurrence_correction.csv>`_
as described above, and will also update `leap/processed_data/occurrence_calibration_parameters.json
<https://github.com/resplab/leap/blob/main/leap/processed_data/occurrence_calibration_parameters.json>`_:

.. code-block:: json

   {
      "β_fhx_age": 0.6445257,
      "β_abx_age": -0.2968535
   }

.. warning::

   Rerunning the beta parameters optimization is slow - could take up to 24 hours.


leap.data\_generation.occurrence\_calibration\_data module
*************************************************************

.. automodule:: leap.data_generation.occurrence_calibration_data
   :members:
   :undoc-members:
   :show-inheritance:
