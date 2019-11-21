{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

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

      .. autosummary::
         :nosignatures:
         :toctree: methods
      
      {{ ([objname] + ["plot"] + ["get_roc_data"]) | join('.') }}

   {% endif %}

   {% endblock %}


