{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {% set class_attributes = ['default_pipelines'] %}

   {% block attributes %}
   .. Class attributes:
   .. autoattribute:: default_pipelines
   
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
