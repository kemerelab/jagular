=======
Jagular
=======

Out-of-core pre-processing of big-ish electrophysiology data, including spike detection and alignment.

Installation
============

The easiest way to install jagular is to use ``pip``. From the terminal, run:

.. code-block:: bash

    pip install jagular

Alternatively, you can install the latest version of jagular by running the following commands:

.. code-block:: bash

    git clone https://github.com/kemerelab/jagular.git
    cd jagular
    python setup.py [install, develop]

where the ``develop`` argument should be used if you want to modify the code.

What is Jagular used for?
=========================
We perform long (multiple days to multiple week-long) chronic in-vivo electrophysilogy recordings, resulting in many terabytes of data per experiment. These long recording periods pose significant challenges to almost every part of a typical analysis pipeline, including filtering, spike detection, and alignment.

For spike detection, for example, we need to filter the wideband data prior to doing some form of threshold crossing detection. But if we have terabytes worth of data, even a simple step such as filtering can become tricky, since we have to do the filtering out-of-core (since the data does not fit into memory). In addition, there can be substantial drift on the electrodes over such a long period of time, so an adaptive threshold crossing aproach would be mre appropriate.

Jagular makes doing these out-of-core tasks easier, by providing a simple interface to read in well-defined chunks from multiple files, in a seamless manner. These chunks can then be used to process the data in a more manageable way. Jagular also has complete support built in for the full (filtering)-(spike-detection)-(waveform-allignment) part of the analysis process, which works out-of-core, and deals elegantly with electrode drift.

Where
=====

===================   ========================================================
 download             https://pypi.python.org/pypi/jagular
 inspiration          https://www.youtube.com/watch?v=WLCDAPNTpaM
 docs                 coming soon!
 code                 https://github.com/kemerelab/jagular
===================   ========================================================

License
=======

Jagular is distributed under the MIT license. See the `LICENSE <https://github.com/kemerelab/jagular/blob/master/LICENSE>`_ file for details.
