API Reference
=============

*******************************************
:doc:`AutoML <autoapi/evalml/automl/index>`
*******************************************
Classes and functions related to the automated search of machine learning pipelines. For a tutorial, see the
:doc:`AutoMLSearch User Guide <user_guide/automl>`

*****************************************************
:doc:`Data Checks <autoapi/evalml/data_checks/index>`
*****************************************************
Classes that can determine if your data is healthy enough to run :doc:`AutoMLSearch <autoapi/evalml/automl/index>`. For
a tutorial, see the :doc:`DataChecks User Guide <user_guide/data_checks>`

*************************************************
:doc:`Demo Datasets <autoapi/evalml/demos/index>`
*************************************************
Functions used to load demo data. This data can be used to try out :doc:`AutoMLSearch <autoapi/evalml/automl/index>`
and our :doc:`pipelines <autoapi/evalml/pipelines/index>`.

**********************************************************
:doc:`Model Families <autoapi/evalml/model_family/index>`
**********************************************************
A categorization of the different kinds of :doc:`Estimators <autoapi/evalml/pipelines/components/estimators/index>` our
:doc:`pipelines <autoapi/evalml/pipelines/index>` support.

*********************************************************************
:doc:`Model Understanding <autoapi/evalml/model_understanding/index>`
*********************************************************************
Functions and classes used to interpret the pipelines found by :doc:`AutoMLSearch <autoapi/evalml/automl/index>`. For a
tutorial, see the :doc:`Model Understanding User Guide <user_guide/model_understanding>`

***************************************************
:doc:`Objectives <autoapi/evalml/objectives/index>`
***************************************************
Classes that represent the cost function :doc:`AutoMLSearch <autoapi/evalml/automl/index>` will optimize. For a tutorial,
see the :doc:`Objectives User Guide <user_guide/objectives>`

*************************************************
:doc:`Pipelines <autoapi/evalml/pipelines/index>`
*************************************************
Classes that represent a sequence of operations to be applied to data. :doc:`AutoMLSearch <autoapi/evalml/automl/index>`
searches over these classes. For a tutorial, see the :doc:`Pipelines User Guide <user_guide/pipelines>`

*********************************************************
:doc:`Preprocessing <autoapi/evalml/preprocessing/index>`
*********************************************************
Functions for splitting and preprocessing data before doing machine learning.

*********************************************************
:doc:`Problem Types <autoapi/evalml/problem_types/index>`
*********************************************************
The different types of machine learning problems that our estimators, pipelines, and AutoMLSearch can handle.

*********************************************************
:doc:`AutoML Tuners <autoapi/evalml/tuners/index>`
*********************************************************
Tuners implement different strategies for sampling from a search space. They’re used in to search the space of pipeline hyperparameters.

*********************************************************
:doc:`Utilities <autoapi/evalml/utils/index>`
*********************************************************
General utilities used primarily by EvalML developers.

**********************************************************
:doc:`Custom Exceptions <autoapi/evalml/exceptions/index>`
**********************************************************
Exceptions that EvalML can raise.

.. toctree::
    :hidden:

    autoapi/evalml/index