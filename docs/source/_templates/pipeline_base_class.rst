{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {% set class_attributes = ['name', 'custom_name', 'summary', 'component_graph', 'problem_type',
                              'model_family', 'hyperparameters', 'custom_hyperparameters',
                              'default_parameters'] %}


   {% block attributes %}
   .. Class attributes:

   .. autoattribute:: name
   .. autoattribute:: custom_name
   .. autoattribute:: problem_type

   {% endblock %}

   {% block instance_attributes %}
   .. rubric:: Instance attributes

   .. autosummary::
      :nosignatures:
      :toctree: attributes

   {% for item in attributes %}
   {% if item not in class_attributes %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {% endfor %}
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

Class Inheritance
"""""""""""""""""

.. inheritance-diagram:: {{ objname }}
