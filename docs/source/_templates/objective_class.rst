{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. inheritance-diagram:: {{ objname }}

.. autoclass:: {{ objname }}
   {% set class_attributes = ['name', 'greater_is_better', 'score_needs_proba'] %}


   {% block attributes %}
   .. Class attributes:

   .. autoattribute:: name
   .. autoattribute:: greater_is_better
   .. autoattribute:: score_needs_proba

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
