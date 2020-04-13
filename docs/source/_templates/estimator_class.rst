{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {% set class_attributes = ['name', 'model_family', 'hyperparameter_ranges', 'supported_problem_types'] %}

   {% block attributes %}
   .. Class attributes:
   .. autoattribute:: name
   .. autoattribute:: model_family
   .. autoattribute:: supported_problem_types
   .. autoattribute:: hyperparameter_ranges
   {% endblock %}

   {% block instance_attributes %}
   .. rubric:: Instance attributes

   .. autosummary::
      :nosignatures:

   {% for item in attributes %}
   {%- if item not in class_attributes %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endblock %}
      
   {% block methods %}
   {% if methods %}
   .. rubric:: Methods:

   .. autosummary::
      :nosignatures:
      :toctree: methods

   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
