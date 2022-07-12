{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {% set class_attributes = ['name'] %}


   {% block attributes %}
   .. Class attributes:

   .. autoattribute:: name

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
