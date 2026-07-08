#pragma once

#include "model/common.h"

// Apply mode-specific env defaults only when the variable is unset (never
// overrides an explicit export). Safe to call from ChatEngine::init.
void apply_dev_mode_env_defaults(DevMode mode, const CParam& param);

// Apply mode-specific CParam defaults for fields the caller did not set.
void apply_dev_mode_cparam_defaults(CParam& param, const CParamUserFlags& flags);
