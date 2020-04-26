{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {% set class_attributes = ['name', 'custom_name', 'summary', 'component_graph', 'problem_type', 'model_family', 'hyperparameters', 'custom_hyperparameters'] %}

   {% block attributes %}
   .. Class attributes:

   .. autoattribute:: name
   .. autoattribute:: custom_name
   .. autoattribute:: summary
   .. autoattribute:: component_graph
   .. autoattribute:: problem_type
   .. autoattribute:: model_family
   .. autoattribute:: hyperparameters
   .. autoattribute:: custom_hyperparameters
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
