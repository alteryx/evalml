{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {% block methods %}
   .. rubric Members
   :members:
   
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
   :attributes:
   {% endblock %}
