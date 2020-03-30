{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {% block members %}
   {% if members %}
   .. rubric:: Members

   .. autosummary::
      :nosignatures:

   {% for item in members %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {%- endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
      :toctree: methods

   {% for item in methods %}
   {%- if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}

   .. rubric:: Attributes
   {% if attributes %}
   .. autosummary::
      :nosignatures:

   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
