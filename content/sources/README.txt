Static MiAi source folder
=========================

No PHP is required.

Files:
- active.csv       : active MiAi v2 chunk CSV
- references.csv   : optional separate reference table
- manifest.json    : points MiAi to both files

Replace active.csv/references.csv directly on the server. Set auto_load=true in
manifest.json if the source should load when MiAi starts.
