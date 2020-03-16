# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['/Users/yuki/OneDrive - The University of Tokyo/imdetector/imdetector/IMDetector.py'],
             pathex=['/Users/yuki/OneDrive - The University of Tokyo/imdetector/imdetector/', '/Users/yuki/OneDrive - The University of Tokyo/imdetector'],
             binaries=[],
             datas=[('model/photopicker_rf_lee_2700.sav', '.'), ('model/photopicker_rf_lee_2700.sav-param.npz', '.')],
             hiddenimports=['matplotlib.backends.backend_macosx', 'sklearn.neighbors._typedefs', 'sklearn.utils._cython_blas', 'sklearn.neighbors._quad_tree', 'sklearn.tree._utils', 'sklearn.ensemble.forest', 'sklearn.tree.tree'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='IMDetector',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='IMDetector')
app = BUNDLE(coll,
             name='IMDetector.app',
             icon=None,
             bundle_identifier=None)
