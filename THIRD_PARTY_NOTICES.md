# Source and license notices

The native toolkit code derives from [lmintlcx/pvztoolkit](https://github.com/lmintlcx/pvztoolkit) (植物大战僵尸 PC 版综合修改器). The toolkit uses GNU GPL version 3; its LICENSE is retained, together with existing source notices.

`native/vendor/miniz` contains the upstream project's miniz compression implementation. Its authorship and license statements remain in the original source files. No FLTK implementation is included; the former FLTK zlib include shim is replaced with direct miniz use.

The newly written retail save serializer uses format information described by the [retail PlayerInfo reconstruction](https://github.com/ruslan831/PlantsVsZombies-decompilation/blob/master/Lawn/System/PlayerInfo.cpp) and store-item definitions; it contains no copied game binary, asset or personal save.

Plants vs. Zombies and its game files are owned by their respective rights holders. Users must provide their own game installation. The source-code license does not grant redistribution rights to the game. Release archives contain this project's compiled native DLL and license notices, with corresponding project source provided separately.
