{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {% set special_methods = ['__str__', '__eq__'] %}


   {% block attributes %}
   .. Class attributes:

   .. autoattribute:: message_type

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

   {% for item in special_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

Class Inheritance
"""""""""""""""""

.. inheritance-diagram:: {{ objname }}
