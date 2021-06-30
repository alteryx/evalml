{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
      :toctree: methods
      :exclude-members: __init__

   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :nosignatures:

   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endblock %}

Class Inheritance
"""""""""""""""""

.. inheritance-diagram:: {{ objname }}
