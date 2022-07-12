{% if obj.display %}
.. py:{{ obj.type }}:: {{ obj.short_name }}{% if obj.args %}({{ obj.args }}){% endif %}
{% for (args, return_annotation) in obj.overloads %}
   {{ " " * (obj.type | length) }}   {{ obj.short_name }}{% if args %}({{ args }}){% endif %}
{% endfor %}


   {% if obj.bases %}
   {% if "show-inheritance" in autoapi_options %}
   Bases: {% for base in obj.bases %}{{ base|link_objs }}{% if not loop.last %}, {% endif %}{% endfor %}
   {% endif %}


   {% if "show-inheritance-diagram" in autoapi_options and obj.bases != ["object"] %}
   .. autoapi-inheritance-diagram:: {{ obj.obj["full_name"] }}
      :parts: 1
      {% if "private-members" in autoapi_options %}
      :private-bases:
      {% endif %}

   {% endif %}
   {% endif %}
   {% if obj.docstring %}
   {{ obj.docstring|prepare_docstring|indent(3) }}
   {% endif %}
   {% if "inherited-members" in autoapi_options %}
   {% set visible_classes = obj.classes|selectattr("display")|rejectattr("name", "equalto", "args")|list %}
   {% else %}
   {% set visible_classes = obj.classes|rejectattr("inherited")|rejectattr("name", "equalto", "args")|selectattr("display")|list %}
   {% endif %}
   {% for klass in visible_classes %}
   {{ klass.render()|indent(3) }}
   {% endfor %}
   {% if "inherited-members" in autoapi_options %}
   {% set visible_attributes = obj.attributes|selectattr("display")|list %}
   {% else %}
   {% set visible_attributes = obj.attributes|rejectattr("inherited")|selectattr("display")|list %}
   {% endif %}

   {% if visible_attributes|length %}
   **Attributes**

   .. list-table::
      :widths: 15 85
      :header-rows: 0

   {% for attribute in visible_attributes|sort(attribute='name') %}
      * - **{{ attribute.name }}**
   {% if attribute.docstring|length > 0 %}
        - {{ attribute.docstring.replace("\n", "") }}
   {% else %}
        - {{ attribute.value|string }}
   {% endif %}
   {% endfor %}
   {% endif %}

   {% if "inherited-members" in autoapi_options %}
   {% set visible_methods = obj.methods|selectattr("display")|rejectattr("name", "equalto", "with_traceback")|list %}
   {% else %}
   {% set visible_methods = obj.methods|rejectattr("inherited")|rejectattr("name", "equalto", "with_traceback")|selectattr("display")|list %}
   {% endif %}

   {% if visible_methods|length %}
   **Methods**

   .. autoapisummary::
      :nosignatures:

   {% for method in visible_methods|sort(attribute='name') %}
      {{ obj.obj["full_name"] }}.{{ method.name }}
   {% endfor %}

   {% for method in visible_methods|sort(attribute='name') %}
   {{ method.render()|indent(3) }}
   {% endfor %}
   {% endif %}
{% endif %}
