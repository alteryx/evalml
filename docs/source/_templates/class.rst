{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {% block instance_attributes %}
   .. rubric:: Instance attributes

   .. autosummary::
      :nosignatures:
      :toctree: attributes

   {% for item in attributes %}
   {%- if item not in class_attributes %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

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
